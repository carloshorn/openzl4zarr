import zarr
from .codec import OpenZLCodec

zarr.codecs.register_codec(OpenZLCodec.name, OpenZLCodec)
