import zarr
from .codec import OpenZLCodec

__version__ = "0.0.1"
__license__ = "MIT"
zarr.codecs.register_codec(OpenZLCodec.name, OpenZLCodec)
