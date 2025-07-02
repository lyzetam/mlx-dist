#!/usr/bin/env python3
import mlx.core as mx
from mlx_lm import load, stream_generate

print("Loading model...")
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")  # Smaller model for testing

prompt = "What is machine learning?"
print(f"\nPrompt: {prompt}\n")
print("Response: ", end="")

for token in stream_generate(model, tokenizer, prompt, max_tokens=100):
    print(token, end='', flush=True)
print("\n\nDone!")