"""
Train the Bigram Language Model v2 - with self-attention!
Compare results to v1 to see the benefit of attention.
"""
import os
import pickle
import torch
import numpy as np

from bigram_v2 import BigramLanguageModelV2

torch.manual_seed(1337)

# Load meta info for encoding/decoding
data_dir = 'shakespeare_char'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# Load the training and validation data
train_data = torch.tensor(np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16).astype(np.int64))
val_data = torch.tensor(np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16).astype(np.int64))

block_size = 8   # context length (must match bigram_v2.py)
batch_size = 32  # how many independent sequences to process in parallel
eval_iters = 200
eval_interval = 500
max_iters = 5000
learning_rate = 1e-3

def get_batch(split: str):
    """Generate a small batch of data of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets, averaged over eval_iters batches."""
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

# Create the model
# v2 has: token embeddings + positional embeddings + self-attention head
# This lets the model actually USE context, not just the last character
m = BigramLanguageModelV2(vocab_size)

# Count and display parameters
n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,}")
print(f"v1 had ~4,225 parameters, v2 has {n_params:,} parameters")
print()

# Create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
print("Training v2 model with self-attention...")
print("v1 baseline converged to ~2.48 loss")
print()

for step in range(max_iters):
    # Periodically evaluate loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = m(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
losses = estimate_loss()
print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
print()

# Generate sample text
print("--- Generated text ---")
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
