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

# Tokenize and write to disk in CHUNKS to avoid OOM!
# Why? 472M Python ints × 28 bytes each ≈ 13GB RAM. Stream to disk instead.
CHUNK_SIZE = 50_000  # stories per chunk

def tokenize_to_file(split, filepath):
    """Tokenize a dataset split and stream to disk in chunks."""
    data = dataset[split]
    n_stories = len(data)
    total_tokens = 0

    with open(filepath, 'wb') as f:
        for start in range(0, n_stories, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_stories)
            chunk_ids = []
            for i in range(start, end):
                chunk_ids.extend(enc.encode_ordinary(data[i]['text']))

            # Convert chunk to uint16 and write immediately, then free memory
            chunk_arr = np.array(chunk_ids, dtype=np.uint16)
            chunk_arr.tofile(f)
            total_tokens += len(chunk_ids)

            print(f"  {end:,} / {n_stories:,} stories ({total_tokens:,} tokens)")

    return total_tokens

print("\nTokenizing training data (streaming to disk)...")
output_dir = os.path.dirname(__file__)
train_path = os.path.join(output_dir, 'train.bin')
n_train = tokenize_to_file('train', train_path)
print(f"Train BPE tokens: {n_train:,}")

print("\nTokenizing validation data...")
val_path = os.path.join(output_dir, 'val.bin')
n_val = tokenize_to_file('validation', val_path)
print(f"Val BPE tokens: {n_val:,}")

# Save to binary files
print("\nSaving metadata...")

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
print(f"Train BPE tokens: {n_train:,}")
print(f"Val BPE tokens: {n_val:,}")
print(f"Total BPE tokens: {n_train + n_val:,}")
print()
print("Comparison to character-level:")
print(f"  Char tokens: ~1,904M  →  BPE tokens: ~{n_train/1e6:.0f}M")
print(f"  Compression ratio: ~{1_904_000_000/n_train:.1f}x fewer tokens")
print(f"  But each token carries more meaning!")
