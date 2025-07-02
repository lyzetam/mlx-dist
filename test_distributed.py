#!/usr/bin/env python3
import mlx.core as mx
import time

def test_distributed():
    # Initialize MPI
    world = mx.distributed.init("mpi")
    rank = world.rank()
    size = world.size()
    
    print(f"Node {rank} of {size} ready!")
    
    # Simple computation test
    local_array = mx.array([rank * 10 + i for i in range(5)])
    print(f"Node {rank} local array: {local_array}")
    
    # Gather all arrays
    all_arrays = mx.distributed.all_gather(local_array)
    
    if rank == 0:
        print(f"\nGathered arrays from all nodes: {all_arrays}")
    
    # Synchronize before exit
    mx.distributed.all_gather(mx.array([rank]))
    
    print(f"Node {rank} completed successfully!")

if __name__ == "__main__":
    test_distributed()