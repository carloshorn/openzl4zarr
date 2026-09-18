import pytest
import zarr
import numpy as np
import openzl.ext as zl
from openzl4zarr import OpenZLCodec
from openzl4zarr.exceptions import FormatVersionError, CompressorFormatWarning
from openzl4zarr.array import open_array_with_compressor


@pytest.fixture
def compressor():
    compressor = zl.Compressor()
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)
    return compressor


@pytest.fixture
def zarray(compressor):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION - 1)
    codec = OpenZLCodec.from_serialized(compressor.serialize())
    codecs = (zarr.codecs.BytesCodec(), codec)
    return zarr.zeros((1000, 1000), chunks=(100, 100), dtype="f8", codecs=codecs)


@pytest.fixture
def store(zarray):
    return zarray.store


def test_open_with_compressor(compressor, store):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION - 2)
    zarray = zarr.open_array(store)
    assert zarray.metadata.codecs[-1].compressor is None
    array = open_array_with_compressor(compressor, store)
    assert array.metadata.codecs[-1].compressor is compressor


def test_open_with_new_compressor_raises_incompatibility_error(compressor, store):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    with pytest.raises(FormatVersionError):
        open_array_with_compressor(compressor, store)


def test_open_with_new_compressor_update_format_version(compressor, store):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    old_format_version = zarr.open_array(store).metadata.codecs[-1].format_version
    zarray = open_array_with_compressor(compressor, store, update_format_version=True)
    new_format_version = zarr.open_array(store).metadata.codecs[-1].format_version
    assert old_format_version < new_format_version


@pytest.mark.filterwarnings("ignore:Compressor format version")
def test_update_format_version_does_not_downgrade(compressor, store):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MIN_FORMAT_VERSION)
    old_format_version = zarr.open_array(store).metadata.codecs[-1].format_version
    zarray = open_array_with_compressor(compressor, store, update_format_version=True)
    new_format_version = zarr.open_array(store).metadata.codecs[-1].format_version
    assert old_format_version == new_format_version
