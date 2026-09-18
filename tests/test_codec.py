import pickle
import pytest
import zarr
import numpy as np
import openzl.ext as zl
from openzl4zarr import OpenZLCodec
from openzl4zarr.exceptions import FormatVersionError, CompressorFormatWarning


@pytest.fixture
def compressor():
    compressor = zl.Compressor()
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)
    return compressor


def test_raise_no_format_no_compressor():
    with pytest.raises(ValueError, match="Neither format version nor a compressor"):
        OpenZLCodec()


def test_format_version_from_parameter():
    format_version = zl.MAX_FORMAT_VERSION - 1
    codec = OpenZLCodec(format_version=format_version)
    assert codec.format_version == format_version


def test_format_version_from_compressor(compressor):
    format_version = zl.MAX_FORMAT_VERSION - 1
    compressor.set_parameter(zl.CParam.FormatVersion, format_version)
    codec = OpenZLCodec(compressor=compressor)
    assert codec.format_version == format_version


def test_max_format_version_for_uninitilized_compressor(compressor):
    codec = OpenZLCodec(compressor=compressor)
    assert codec.format_version == zl.MAX_FORMAT_VERSION


def test_raise_incompatible_parameter_format_version(compressor):
    format_version = zl.MAX_FORMAT_VERSION + 1
    with pytest.raises(FormatVersionError, match="Incompatible format version"):
        OpenZLCodec(format_version=format_version)


def test_warn_about_old_compressor_format_version(compressor):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MIN_FORMAT_VERSION)
    with pytest.warns(CompressorFormatWarning, match="consider re-training"):
        OpenZLCodec(compressor=compressor)


def test_raise_compressor_incompatible_with_array_format_version(compressor):
    compressor.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    format_version = zl.MAX_FORMAT_VERSION - 1
    codec = OpenZLCodec(format_version=format_version)
    with pytest.raises(FormatVersionError):
        codec.replace_compressor(compressor)


def test_state_transfer(compressor):
    codec_old = OpenZLCodec(compressor=compressor)
    pickled = pickle.dumps(codec_old)
    codec_new = pickle.loads(pickled)
    assert codec_old.format_version == codec_new.format_version
    codec_old.compressor.serialize() == codec_new.compressor.serialize()
