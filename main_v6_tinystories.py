"""
Train GPT Language Model v6 on TinyStories with BPE tokenization!

The big upgrade: from character-level to GPT-2 BPE tokens.
Each token is a word or subword instead of a single character.

Requires: python tinystories_bpe/prepare.py  (run first!)
"""
import os
import pickle
import torch
import numpy as np
import time
import math
import tiktoken

from bigram_v6 import GPTLanguageModel, block_size

torch.manual_seed(1337)

# Load meta info
data_dir = 'tinystories_bpe'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']

# Use tiktoken for decoding
enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

print(f"Dataset: TinyStories (BPE)")
print(f"Vocab size: {vocab_size:,} (GPT-2 BPE)")

# Load data - memory map for large files
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")
print()

# Training hyperparameters
batch_size = 32         # Reduced from 64 - larger block_size uses more memory
eval_iters = 200
eval_interval = 500
max_iters = 20000

# Learning rate schedule
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 500


def get_lr(step):
    """Learning rate with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    decay_steps = max_iters - warmup_steps
    step_in_decay = step - warmup_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
    
    return min_lr + (max_lr - min_lr) * cosine_decay


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# Create model
m = GPTLanguageModel(vocab_size)

n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Tokens per parameter: {len(train_data) / n_params:.0f}x")
print(f"Context window: {block_size} BPE tokens ≈ {block_size * 4} characters")
print()

optimizer = torch.optim.AdamW(m.parameters(), lr=max_lr)

print("Training v6 (BPE) on TinyStories...")
print(f"LR schedule: warmup {warmup_steps} steps, then cosine decay")
print(f"Previous best: v5 char-level = 1.08 val loss")
print()

start_time = time.time()

for step in range(max_iters):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e} [{elapsed:.1f}s]")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
print()

# Generate text - BPE produces real words!
print("--- Generated text ---")
context = torch.zeros((1, 1), dtype=torch.long)
generated_ids = m.generate(context, max_new_tokens=200)[0].tolist()
print(decode(generated_ids))
