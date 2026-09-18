import warnings
import json
from typing import Any
from collections.abc import Mapping

from zarr.types import AnyArray
import openzl.ext as zl
import zarr

from .codec import OpenZLCodec
from .sharding import OpenZLShardingCodec


def _update_format_version(zarray: AnyArray, format_version: int) -> AnyArray:
    """We assume this is a zarr v3 array"""
    codec = zarray.metadata.inner_codecs[-1]
    if codec.format_version < format_version:
        store = zarray.store
        key = f"{zarray.path}/zarr.json".strip("/")
        buffer = zarr.api.synchronous.sync(store.get(key))
        metadata = json.loads(buffer.to_bytes().decode("utf-8"))
        codecs = metadata["codecs"]
        if codecs[-1]["name"] == "sharding_indexed":
            codecmeta = codecs[-1]["configuration"]["codecs"][-1]
        else:
            codecmeta = codecs[-1]
        codecmeta["configuration"]["format_version"] = format_version
        raw_bytes = json.dumps(metadata, indent=4).encode("utf-8")
        value = buffer.from_bytes(raw_bytes)
        zarr.api.synchronous.sync(store.set(key, value))
        array = zarr.open_array(store, path=zarray.path)
    else:
        array = zarray
    return array


def open_array_with_compressor(
    compressor: zl.Compressor, *args, update_format_version: bool = False, **kwargs
) -> AnyArray:
    zarray = zarr.open_array(*args, **kwargs)
    codec = zarray.metadata.inner_codecs[-1]
    if not isinstance(codec, OpenZLCodec):
        raise ValueError("This zarr array does not seem to use OpenZL")
    if update_format_version:
        format_version = compressor.get_parameter(zl.CParam.FormatVersion)
        zarray = _update_format_version(zarray, format_version)
    codec = zarray.metadata.inner_codecs[-1]
    codec.replace_compressor(compressor)
    return zarray


def open_array_with_codec_mapping(
    open_array_with_codec_mapping: Mapping[tuple[int, ...], OpenZLCodec],
    *args,
    update_format_version: bool = False,
    **kwargs,
) -> AnyArray:
    with zarr.config.set(
        {"codecs.sharding_indexed": "openzl4zarr.sharding.OpenZLShardingCodec"}
    ):
        zarray = zarr.open_array(*args, **kwargs)
        if not isinstance(zarray.metadata.codecs[-1], OpenZLShardingCodec):
            raise ValueError("This zarr array does not seem to use Sharding")
        if update_format_version:
            format_version = (
                max(
                    (codec.format_version for codec in codec_mapping.values()),
                    default=0,
                )
                or zl.MAX_FORMAT_VERSION
            )
            zarray = _update_format_version(zarray, foramt_version)
    zarray.metadata.codecs[-1].set_codec_mapping(codec_mapping)
    return zarray
