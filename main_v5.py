"""
Train the Bigram Language Model v5 - Low-hanging fruit optimizations!

New in v5:
1. GELU activation (in model)
2. Weight tying (in model)
3. Learning rate warmup + cosine decay (here!)

The learning rate schedule:
- Warmup: Start low, ramp up over first 500 steps
- Cosine decay: Gradually decrease following a cosine curve
- Min LR: Don't go below 10% of max LR
"""
import os
import pickle
import torch
import numpy as np
import time
import math

from bigram_v5 import BigramLanguageModelV5, block_size

torch.manual_seed(1337)

# Load meta info
data_dir = 'shakespeare_char'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# Load data
train_data = torch.tensor(np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16).astype(np.int64))
val_data = torch.tensor(np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16).astype(np.int64))

# Training hyperparameters
batch_size = 64
eval_iters = 200
eval_interval = 500
max_iters = 10000

# Learning rate schedule parameters
max_lr = 3e-4          # peak learning rate
min_lr = max_lr * 0.1  # minimum learning rate (10% of max)
warmup_steps = 500     # linear warmup for this many steps


def get_lr(step):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Visualized:
        LR
        ↑
        |      /‾‾‾‾‾\
        |     /       \
        |    /         \
        |   /           \____
        |__/
        └────────────────────→ Steps
        warmup    cosine decay
    """
    # 1. Warmup phase: linear increase from 0 to max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2. After warmup: cosine decay from max_lr to min_lr
    decay_steps = max_iters - warmup_steps
    step_in_decay = step - warmup_steps
    
    # Cosine decay formula
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
    
    return min_lr + (max_lr - min_lr) * cosine_decay


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
m = BigramLanguageModelV5(vocab_size)

n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"v4 had 1,212,481 params - we saved {1212481 - n_params:,} from weight tying!")
print()

# Optimizer (we'll update LR manually each step)
optimizer = torch.optim.AdamW(m.parameters(), lr=max_lr)

print("Training v5 model (GELU + weight tying + LR schedule)...")
print(f"LR schedule: warmup {warmup_steps} steps, then cosine decay")
print(f"  max_lr={max_lr}, min_lr={min_lr}")
print()
print("Previous results:")
print("  v3: ~1.77 loss | v4: ~1.56 loss | v5: ???")
print()

start_time = time.time()

for step in range(max_iters):
    # Update learning rate based on schedule
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
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
