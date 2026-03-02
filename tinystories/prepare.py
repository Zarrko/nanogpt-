"""
Prepare the TinyStories dataset for character-level language modeling.
TinyStories is a dataset of ~2.5M short stories written for children.
Much larger than Shakespeare (~500MB vs ~1MB), great for training larger models.

Downloads from HuggingFace: roneneldan/TinyStories
Saves train.bin, val.bin (tokenized) and meta.pkl (vocab mappings).

Usage:
    pip install datasets  # if not already installed
    python tinystories/prepare.py
"""
import os
import pickle
import numpy as np

# Try to import datasets, give helpful error if not installed
try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library:")
    print("  pip install datasets")
    exit(1)

print("Loading TinyStories dataset from HuggingFace...")
print("(This may take a few minutes on first run - downloading ~500MB)")

# Load the dataset - uses 'train' and 'validation' splits
dataset = load_dataset("roneneldan/TinyStories")

print(f"Train stories: {len(dataset['train']):,}")
print(f"Validation stories: {len(dataset['validation']):,}")

# Combine all stories into single strings
print("\nCombining stories into text...")
train_text = "\n\n".join([story['text'] for story in dataset['train']])
val_text = "\n\n".join([story['text'] for story in dataset['validation']])

print(f"Train text length: {len(train_text):,} characters")
print(f"Val text length: {len(val_text):,} characters")

# Build vocabulary from training data only (to avoid data leakage)
print("\nBuilding vocabulary...")
chars = sorted(list(set(train_text)))
# Also add any chars that only appear in validation (rare but possible)
val_chars = set(val_text) - set(chars)
if val_chars:
    print(f"Adding {len(val_chars)} chars from validation set")
    chars = sorted(list(set(train_text) | set(val_text)))

vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {repr(''.join(chars[:50]))}..." if len(chars) > 50 else f"Characters: {repr(''.join(chars))}")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Convert string to list of integers."""
    return [stoi[c] for c in s]

def decode(l):
    """Convert list of integers to string."""
    return ''.join([itos[i] for i in l])

# Encode the data
print("\nEncoding text to tokens...")
train_ids = encode(train_text)
val_ids = encode(val_text)
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")

# Save to binary files
print("\nSaving to binary files...")
output_dir = os.path.dirname(__file__)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_path = os.path.join(output_dir, 'train.bin')
val_path = os.path.join(output_dir, 'val.bin')
train_ids.tofile(train_path)
val_ids.tofile(val_path)
print(f"Saved {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
print(f"Saved {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")

# Save meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
print(f"Saved {meta_path}")

# Summary
print("\n" + "=" * 50)
print("TinyStories dataset prepared!")
print("=" * 50)
print(f"Vocabulary size: {vocab_size}")
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")
print(f"Total tokens: {len(train_ids) + len(val_ids):,}")
print(f"\nTokens per parameter ratio (at 1.2M params): {len(train_ids) / 1_200_000:.1f}x")
print("(Compare to Shakespeare: ~0.75x - now we have MUCH more data!)")
