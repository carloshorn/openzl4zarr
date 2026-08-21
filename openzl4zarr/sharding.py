"""Module containing the OpenZLShardingCodec."""
from openzl4zarr import OpenZLCodec
import openzl.ext as zl
import zarr
from zarr.codecs.sharding import *

from dataclasses import field

if TYPE_CHECKING or True:
    from collections.abc import Iterator, Mapping
    from typing import Self, Optional

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


@dataclass(frozen=True)
class OpenZLShardingCodec(zarr.codecs.ShardingCodec):
    """OpenZL sharding codec enables using different OpenZLCodec instances on each inner chunk.

    `codec_mapping` maps chunk coordinate to OpenZLCodec instances.
    """
    codec_mapping: Mapping[tuple[int, ...], OpenZLCodec] = field(default_factory=dict)

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        index_codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(), Crc32cCodec()),
        index_location: ShardingCodecIndexLocation | str = "end",
        subchunk_write_order: SubchunkWriteOrder = "morton",
        codec_mapping: Optional[Mapping[tuple[int, ...], OpenZLCodec]] = None
    ) -> None:
        codecs = (BytesCodec(), OpenZLCodec())
        super().__init__(
            chunk_shape=chunk_shape, codecs=codecs, index_codecs=index_codecs, 
            index_location=index_location, subchunk_write_order=subchunk_write_order
        )
        if codec_mapping is None:
            codec_mapping = dict()
        object.__setattr__(self, "codec_mapping", codec_mapping)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["codec_mapping"] = self.codec_mapping
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        object.__setattr__(self, "codec_mapping", state["codec_mapping"])

    def _encode_sync(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
    ) -> Buffer | None:
        """Encode a full shard synchronously.

        Sync counterpart to `_encode_single`. This is reached when a
        `ShardingCodec` is an *inner* codec of another sharding codec (nested
        sharding): the outer codec encodes each inner chunk through its
        `ChunkTransform`, which calls this method on the inner `ShardingCodec`.

        Each inner chunk is encoded through the inner `ChunkTransform` and
        collected into an intermediate `dict`. The dict's key order is
        immaterial — the physical on-disk layout is decided downstream by the
        `subchunk_write_order` loop in `_encode_shard_dict_sync` (this method
        does NOT impose a layout). Empty inner chunks become `None` entries when
        `write_empty_chunks` is False, signalling `_encode_shard_dict_sync` to
        elide them from the data section and mark them empty in the shard index.

        Returns `None` if every inner chunk was elided (an all-empty shard) —
        callers treat that as "delete the shard key".

        This method does not parallelize compression, but should.
        See TODO: make issue for handling subchunk parallelism

        For a partial write that only touches some inner chunks, use
        `_encode_partial_sync` instead.
        """
        shard_shape = shard_spec.shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=ChunkGrid.from_sizes(shard_shape, self.chunk_shape),
        )

        # Key order here is immaterial; _encode_shard_dict_sync lays the present
        # chunks out in subchunk_write_order.
        shard_builder: dict[tuple[int, ...], Buffer | None] = dict.fromkeys(
            lexicographic_order_coords(chunks_per_shard)
        )

        for chunk_coords, _chunk_selection, out_selection, _ in indexer:
            # None = chunk normalized to missing (see encode_or_elide_chunk)
            inner_transform = self._get_located_inner_chunk_transform(shard_spec, chunk_coords)
            shard_builder[chunk_coords] = encode_or_elide_chunk(
                shard_array[out_selection], chunk_spec, inner_transform.encode_chunk
            )

        return self._encode_shard_dict_sync(
            shard_builder,
            chunks_per_shard=chunks_per_shard,
            buffer_prototype=default_buffer_prototype(),
        )

    def _get_located_inner_chunk_transform(self, shard_spec, chunk_coords):
        chunk_spec = self._get_chunk_spec(shard_spec)
        openzl_codec = self.codec_mapping[chunk_coords]
        codecs = self.codecs[:-1] + (openzl_codec,)
        return ChunkTransform(codecs=evolve_codecs(codecs, chunk_spec))

    async def _encode_partial_single(
        self,
        byte_setter: ByteSetter,
        shard_array: NDBuffer,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = list(
            get_indexer(
                selection,
                shape=shard_shape,
                chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
            )
        )

        if not self._is_complete_shard_write(indexer, chunks_per_shard):
            raise RuntimeError("Not supported in this POC!")
        buf = self._encode_sync(shard_array, shard_spec)
        if buf is None:
            await byte_setter.delete()
        else:
            await byte_setter.set(buf)
