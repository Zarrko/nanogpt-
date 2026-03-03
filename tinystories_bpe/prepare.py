"""
Prepare the TinyStories dataset with BPE (Byte Pair Encoding) tokenization.

Character-level: "Hello" → ['H', 'e', 'l', 'l', 'o'] → 5 tokens
BPE (GPT-2):     "Hello" → ['Hello']                  → 1 token!

BPE groups common character sequences into single tokens.
GPT-2's tokenizer has 50,257 tokens (vs our 174 characters).

Advantage: Fewer tokens per sentence = more context per block_size
Disadvantage: Much larger vocabulary = bigger embedding table

Downloads TinyStories from HuggingFace and tokenizes with tiktoken (GPT-2 BPE).
Saves train.bin, val.bin, and meta.pkl.

Usage:
    pip install datasets tiktoken
    python tinystories_bpe/prepare.py
"""
import os
import pickle
import numpy as np

# Try imports, give helpful errors
try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install datasets")
    exit(1)

try:
    import tiktoken
except ImportError:
    print("Please install: pip install tiktoken")
    exit(1)

print("Loading TinyStories dataset from HuggingFace...")
print("(This may take a few minutes on first run)")

dataset = load_dataset("roneneldan/TinyStories")

print(f"Train stories: {len(dataset['train']):,}")
print(f"Validation stories: {len(dataset['validation']):,}")

# Load GPT-2 BPE tokenizer
print("\nLoading GPT-2 BPE tokenizer...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50,257
print(f"BPE vocabulary size: {vocab_size:,}")

# Show example comparison
example = "Once upon a time, there was a little girl named Lily."
char_tokens = list(example)
bpe_tokens = enc.encode(example)
print(f"\nExample: {example!r}")
print(f"  Character tokens: {len(char_tokens)} tokens")
print(f"  BPE tokens: {len(bpe_tokens)} tokens ({len(char_tokens)/len(bpe_tokens):.1f}x compression)")
print(f"  BPE decoded: {[enc.decode([t]) for t in bpe_tokens]}")

# Tokenize all stories
print("\nTokenizing training data with BPE (this may take a while)...")
train_ids = []
for i, story in enumerate(dataset['train']):
    ids = enc.encode_ordinary(story['text'])
    train_ids.extend(ids)
    if (i + 1) % 500_000 == 0:
        print(f"  Processed {i+1:,} / {len(dataset['train']):,} train stories...")

print(f"Train BPE tokens: {len(train_ids):,}")

print("\nTokenizing validation data with BPE...")
val_ids = []
for story in dataset['validation']:
    ids = enc.encode_ordinary(story['text'])
    val_ids.extend(ids)

print(f"Val BPE tokens: {len(val_ids):,}")

# Save to binary files
print("\nSaving to binary files...")
output_dir = os.path.dirname(__file__)

# Use uint16 - GPT-2 vocab is 50,257 which fits in uint16 (max 65,535)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_path = os.path.join(output_dir, 'train.bin')
val_path = os.path.join(output_dir, 'val.bin')
train_ids.tofile(train_path)
val_ids.tofile(val_path)
print(f"Saved {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
print(f"Saved {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")

# Save meta - for BPE we use tiktoken's encode/decode directly
meta = {
    'vocab_size': vocab_size,
    'tokenizer': 'gpt2',  # tells training script to use tiktoken
}
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
print(f"Saved {meta_path}")

# Summary
print("\n" + "=" * 60)
print("TinyStories BPE dataset prepared!")
print("=" * 60)
print(f"Tokenizer: GPT-2 BPE ({vocab_size:,} tokens)")
print(f"Train BPE tokens: {len(train_ids):,}")
print(f"Val BPE tokens: {len(val_ids):,}")
print(f"Total BPE tokens: {len(train_ids) + len(val_ids):,}")
print()
print("Comparison to character-level:")
print(f"  Char tokens: ~1,904M  →  BPE tokens: ~{len(train_ids)/1e6:.0f}M")
print(f"  Compression ratio: ~{1_904_000_000/len(train_ids):.1f}x fewer tokens")
print(f"  But each token carries more meaning!")
