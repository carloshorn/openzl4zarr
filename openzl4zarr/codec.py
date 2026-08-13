"""This module contains the OpenZL codec for Zarr"""
from __future__ import annotations

import asyncio
import pathlib
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import openzl.ext as zl
from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import Buffer as CpuBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self, Optional
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


OPENZL_VERSION = (zl.LIBRARY_VERSION_MAJOR, zl.LIBRARY_VERSION_MINOR, zl.LIBRARY_VERSION_PATCH)

def build_compressor(chunk_spec: ArraySpec):
    """build a generic compressor for the incoming data"""
    compressor = zl.Compressor()
    endianness = getattr(chunk_spec.dtype, "endianness", None)
    if endianness == "little":
        endianness = "LE"
    elif endianness == "big":
        endianness = "BE"
    else:
        endianness = ""
    itemsize = chunk_spec.dtype.item_size * 8
    node = getattr(zl.nodes, f"ConvertSerialToNum{endianness}{itemsize}", None)
    if node is None:
        graph = zl.graphs.Compress()(compressor)
    else:
        graph = node()(compressor, successor=zl.graphs.Compress())
    compressor.select_starting_graph(graph)    
    return compressor


@dataclass(frozen=True)
class OpenZLCodec(BytesBytesCodec):
    """OpenZL codec"""

    is_fixed_size = False
    name = "openzl"

    compressor: Optional[zl.Compressor] = None
    compressor_factory: Callable[[ArraySpec], zl.Compressor] = build_compressor

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, cls.name)
        version = tuple(map(int, configuration_parsed["version"].split(".")))
        if version > OPENZL_VERSION:
            warnings.warn("The data were compressed using a newer version of OpenZL!", RuntimeWarning)
        return cls()

    @classmethod
    def from_path(cls, path):
        path = pathlib.Path(path)
        serialized = path.read_bytes()
        return cls.from_serialized(serialized)

    @classmethod
    def from_serialized(cls, serialized):
        compressor = zl.Compressor()
        compressor.deserialize(serialized)
        return cls(compressor=compressor)

    def to_dict(self) -> dict[str, JSON]:
        version = ".".join(map(str, OPENZL_VERSION))
        return {"name": self.name, "configuration": {"version": version}}

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """decompress a single chunk"""
        # TODO: need to find a way to implement the buffer protocol for decompressor
        # we should not need this type casting which is a copy operation!
        compressed = chunk_bytes.to_bytes()

        dctx = zl.DCtx()
        outputs = dctx.decompress(compressed)
        if len(outputs) != 1 or outputs[0].type != zl.Type.Serial:
            raise RuntimeError("Only one serial output supported")

        chunk_data = outputs[0].content.as_nparray()
        return chunk_spec.prototype.buffer.from_array_like(chunk_data)

    def _encode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        if self.compressor is None:
            warnings.warn("Using an unoptimized default compressor!", RuntimeWarning)
            compressor = self.compressor_factory(chunk_spec)
        else:
            compressor = self.compressor
        cctx = zl.CCtx()
        cctx.ref_compressor(compressor)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        data = chunk_data.as_array_like()
        compressed = cctx.compress([zl.Input(zl.Type.Serial, data)])
        return chunk_spec.prototype.buffer.from_bytes(compressed)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_data, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
