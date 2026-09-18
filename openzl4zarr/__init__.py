from zarr.registry import register_codec
from .codec import OpenZLCodec
from .sharding import OpenZLShardingCodec

__version__ = "0.1.0"
__license__ = "MIT"


register_codec(OpenZLCodec.name, OpenZLCodec)
register_codec("sharding_indexed", OpenZLShardingCodec)
