"""
Train the Bigram Language Model v5 on TinyStories dataset!

TinyStories is ~2000x larger than Shakespeare:
- Shakespeare: ~900k tokens
- TinyStories: ~1.9 BILLION tokens

With this much data, the v5 optimizations (GELU, weight tying, LR schedule)
should actually help - unlike on tiny Shakespeare where they hurt!
"""
import os
import pickle
import torch
import numpy as np
import time
import math

from bigram_v5 import BigramLanguageModelV5, block_size

torch.manual_seed(1337)

# Load meta info from TinyStories
data_dir = 'tinystories'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Dataset: TinyStories")
print(f"Vocab size: {vocab_size} (vs Shakespeare's 65)")

# Load data - memory map for large files!
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")
print()

# Training hyperparameters
batch_size = 64
eval_iters = 200
eval_interval = 500
max_iters = 20000  # Same as Shakespeare run for comparison

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
    # Convert from memmap to tensor
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


# Create model with TinyStories vocab
m = BigramLanguageModelV5(vocab_size)

n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Tokens per parameter: {len(train_data) / n_params:.0f}x (vs Shakespeare's ~0.75x)")
print()

optimizer = torch.optim.AdamW(m.parameters(), lr=max_lr)

print("Training v5 on TinyStories...")
print(f"LR schedule: warmup {warmup_steps} steps, then cosine decay")
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

print("--- Generated text ---")
# Start with newline (common story start)
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
