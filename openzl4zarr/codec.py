"""This module contains the OpenZL codec for Zarr"""

from __future__ import annotations

import asyncio
import pathlib
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import openzl.ext as zl
from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import parse_named_configuration

from .exceptions import CompressorFormatWarning, FormatVersionError

if TYPE_CHECKING:
    from typing import Self, Optional
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON


def _check_compressor(compressor: zl.Compressor) -> None:
    format_version = (
        compressor.get_parameter(zl.CParam.FormatVersion) or zl.MAX_FORMAT_VERSION
    )
    mid_version = (zl.MIN_FORMAT_VERSION + zl.MAX_FORMAT_VERSION) // 2
    if format_version < mid_version:
        warnings.warn(
            f"Compressor format version {format_version} is getting old: "
            f"[min, max] = [{zl.MIN_FORMAT_VERSION}, {zl.MAX_FORMAT_VERSION}] "
            "consider re-training.",
            CompressorFormatWarning,
        )


@dataclass(frozen=True)
class OpenZLCodec(BytesBytesCodec):
    """OpenZL codec"""

    is_fixed_size = False
    name = "openzl4zarr.openzl"

    format_version: int | None = None
    compressor: zl.Compressor | None = None

    def __init__(
        self,
        *,
        format_version: int | None = None,
        compressor: zl.Compressor | None = None,
    ) -> None:
        if format_version is None and compressor is None:
            raise ValueError("Neither format version nor a compressor given!")
        if format_version is None:
            # if not set yet, it is zero, which evaluate to False as bool
            format_version = (
                compressor.get_parameter(zl.CParam.FormatVersion)
                or zl.MAX_FORMAT_VERSION
            )
        if format_version > zl.MAX_FORMAT_VERSION:
            raise FormatVersionError(
                f"Incompatible format version: {format_version} > {zl.MAX_FORMAT_VERSION}"
            )
        object.__setattr__(self, "format_version", format_version)
        self.replace_compressor(compressor)

    def __getstate__(self) -> dict[str, Any]:
        if self.compressor is None:
            serialized = None
        else:
            serialized = self.compressor.serialize()
        return {"compressor": serialized, **self.to_dict()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        config = state["configuration"]
        object.__setattr__(self, "format_version", config["format_version"])
        serialized = state["compressor"]
        if serialized is None:
            compressor = None
        else:
            compressor = zl.Compressor()
            compressor.deserialize(serialized)
        self.replace_compressor(compressor)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, cls.name)
        format_version = configuration_parsed["format_version"]
        return cls(format_version=format_version)

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
        config = {"format_version": self.format_version}
        return {"name": self.name, "configuration": config}

    def replace_compressor(self, compressor: zl.Compressor | None) -> None:
        if compressor is not None:
            _check_compressor(compressor)
            format_version = compressor.get_parameter(zl.CParam.FormatVersion)
            if format_version > self.format_version:
                raise FormatVersionError(
                    "The given compressor has a higher format version than supported! "
                    "You could increase the format version in the array metadata, but "
                    "bear in mind that all client libraries must be updated first."
                )
        object.__setattr__(self, "compressor", compressor)

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
            raise RuntimeError("Need a compressor!")
        compressor = self.compressor
        cctx = zl.CCtx()
        cctx.ref_compressor(compressor)
        cctx.set_parameter(zl.CParam.FormatVersion, self.format_version)
        data = chunk_data.as_array_like()
        compressed = cctx.compress([zl.Input(zl.Type.Serial, data)])
        return chunk_spec.prototype.buffer.from_bytes(compressed)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_data, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError
