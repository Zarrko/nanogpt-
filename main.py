"""
Test the Bigram Language Model.
"""
import os
import pickle
import torch
import numpy as np

from bigram import BigramLanguageModel

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

block_size = 8  # maximum context length for predictions (how many tokens the model looks at to predict the next one)
batch_size = 4  # how many independent sequences to process in parallel

def get_batch(split: str):
    """Generate a small batch of data of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

eval_iters = 200

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
# ~4,225 parameters (65 vocab × 65 embedding) - just a single lookup table!
# Bigram = only looks at 1 previous character. Despite having block_size=8,
# the model ignores the context! It only uses the *last* token to predict the
# next one (that's what bigram means). This is why it runs fine on CPU.
m = BigramLanguageModel(vocab_size)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
eval_interval = 1000
for steps in range(10000):
    # periodically evaluate loss on train and val sets
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Each step the model sees batch_size * block_size training examples
    # Over 10k steps: ~320k positions, learning which characters follow others

    # sample a batch of data
    # randomly grabs batch_size chunks of block_size characters from Shakespeare
    xb, yb = get_batch('train')

    # forward pass: predict next character, compute loss (how wrong we are)
    # loss starts ~4.2 (random guess for 65 chars) and should drop toward ~2.5
    logits, loss = m(xb, yb)

    # clear old gradients from previous step
    optimizer.zero_grad(set_to_none=True)

    # backpropagation: compute how much each weight contributed to the error
    loss.backward()

    # weight update: adjust weights slightly to reduce loss
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
