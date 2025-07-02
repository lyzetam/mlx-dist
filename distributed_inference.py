#!/usr/bin/env python3
import mlx.core as mx
from mlx_lm import load, stream_generate
import time
import psutil
import os

def log_system_stats(rank):
    """Log system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"[Node {rank}] CPU: {cpu_percent}%, Memory: {memory.percent}%")

def distributed_inference_with_monitoring():
    try:
        # Initialize MPI backend
        world = mx.distributed.init(backend="mpi")
        rank = world.rank()
        size = world.size()
        
        if rank == 0:
            print("="*50)
            print(f"MLX Distributed Inference Starting")
            print(f"Master: {os.uname().nodename} ({rank})")
            print(f"Total nodes: {size}")
            print("="*50)
        
        # Log initial system stats
        log_system_stats(rank)
        
        # Load model with timing
        start_time = time.time()
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        load_time = time.time() - start_time
        
        print(f"[Node {rank}] Model loaded in {load_time:.2f}s")
        
        # Synchronize before inference
        mx.distributed.all_gather(mx.array([rank]))
        
        if rank == 0:
            # Master node handles the prompt
            prompts = [
                "What are the benefits of distributed computing?",
                "Explain machine learning in one paragraph.",
                "How does Apple Silicon optimize ML workloads?"
            ]
            
            for i, prompt in enumerate(prompts):
                print(f"\n{'='*50}")
                print(f"Query {i+1}: {prompt}")
                print(f"{'='*50}\n")
                
                start_time = time.time()
                tokens_generated = 0
                
                for token in stream_generate(
                    model, tokenizer, prompt,
                    max_tokens=200,
                    temp=0.7
                ):
                    print(token, end='', flush=True)
                    tokens_generated += 1
                
                inference_time = time.time() - start_time
                tokens_per_second = tokens_generated / inference_time
                
                print(f"\n\n[Performance] Generated {tokens_generated} tokens")
                print(f"[Performance] Time: {inference_time:.2f}s")
                print(f"[Performance] Speed: {tokens_per_second:.1f} tokens/s")
                
                # Log system stats after each query
                log_system_stats(rank)
        
        else:
            # Worker nodes participate in generation
            for _ in range(3):  # Match number of prompts
                for _ in stream_generate(
                    model, tokenizer, "",
                    max_tokens=200,
                    temp=0.7
                ):
                    pass
                log_system_stats(rank)
        
        if rank == 0:
            print("\n" + "="*50)
            print("Distributed inference completed successfully!")
            print("="*50)
            
    except Exception as e:
        print(f"[Node {rank}] Error: {e}")
        mx.distributed.abort()

if __name__ == "__main__":
    distributed_inference_with_monitoring()