"""
Train the Bigram Language Model v3 - Multi-head + stacked layers!
Should produce much better text than v1/v2.
"""
import os
import pickle
import torch
import numpy as np

from bigram_v3 import BigramLanguageModelV3, block_size

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
batch_size = 32
eval_iters = 200
eval_interval = 500
max_iters = 5000
learning_rate = 1e-3

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
m = BigramLanguageModelV3(vocab_size)

n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"v1: ~4k params, v2: ~5k params, v3: {n_params:,} params")
print()

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("Training v3 model (multi-head + layers)...")
print("v1 baseline: ~2.48 loss")
print("v2 baseline: ~2.41 loss")
print()

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
print()

print("--- Generated text ---")
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
