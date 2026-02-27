"""
Train the Bigram Language Model v4 - Pushing CPU limits!

Larger model, longer training. Should produce much better text.
Expected training time: ~10-15 minutes on CPU.
"""
import os
import pickle
import torch
import numpy as np
import time

from bigram_v4 import BigramLanguageModelV4, block_size

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

# Training hyperparameters - scaled up
batch_size = 64          # larger batches (v3: 32)
eval_iters = 200
eval_interval = 500
max_iters = 10000        # more training steps (v3: 5000)
learning_rate = 3e-4     # slightly lower lr for larger model

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
m = BigramLanguageModelV4(vocab_size)

n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"v1: ~4k | v2: ~5k | v3: ~210k | v4: {n_params//1000}k")
print()

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("Training v4 model (pushing CPU limits)...")
print("v1: ~2.48 | v2: ~2.41 | v3: ~1.77 | v4: ???")
print()

start_time = time.time()

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} [{elapsed:.1f}s]")

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
