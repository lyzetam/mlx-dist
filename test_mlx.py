import mlx.core as mx
from mlx_lm import load, stream_generate
import os


world = mx.distributed.init("mpi")
print(f"Node {world.rank()} of {world.size()} ready!")