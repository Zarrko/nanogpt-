# NanoGPT - Learning Transformers from Scratch

A minimal implementation for understanding GPT-style language models, following Andrej Karpathy's approach.

## Project Structure

```
nanogpt/
├── main.py              # Training script for bigram v1
├── main_v2.py           # Training script for bigram v2 (with attention)
├── main_v3.py           # Training script for bigram v3 (multi-head + layers)
├── bigram.py            # Bigram language model v1
├── bigram_v2.py         # Bigram language model v2 (with self-attention)
├── bigram_v3.py         # Bigram language model v3 (full transformer!)
├── attention.py         # Step-by-step attention implementation
├── shakespeare_char/    # Character-level Shakespeare dataset
│   ├── prepare.py       # Downloads and tokenizes data
│   ├── input.txt        # Raw Shakespeare text
│   ├── train.bin        # Tokenized training data
│   ├── val.bin          # Tokenized validation data
│   └── meta.pkl         # Vocab mappings (stoi, itos)
└── requirements.txt     # Dependencies
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Prepare the Shakespeare dataset
python shakespeare_char/prepare.py

# Train the bigram model
python main.py

# Train the v2 model with self-attention
python main_v2.py

# Train the v3 model (multi-head + layers) - the good one!
python main_v3.py
```

## What We've Built

### 1. Bigram Language Model (`bigram.py`)

The simplest neural language model - predicts the next character based only on the current character.

**Architecture:**
- Single embedding table: 65 × 65 = **4,225 parameters**
- Input: current character → Output: probability distribution over next character

**Key insight:** Despite training on sequences of 8 characters (`block_size=8`), the bigram model ignores context! It only uses the last token. This is the baseline to beat.

**Results:**
- Loss starts at ~4.7 (random: ln(65) ≈ 4.17)
- Converges to ~2.48 (best possible for bigram on Shakespeare)
- Perplexity: e^2.48 ≈ 12 (confused between ~12 characters on average)

### 2. Bigram Language Model v2 (`bigram_v2.py`)

Adds self-attention so the model actually uses context instead of ignoring it!

**Architecture:**
- Token embeddings: 65 × 32 = 2,080 parameters
- Positional embeddings: 8 × 32 = 256 parameters
- Self-attention head (Q, K, V): 3 × (32 × 16) = 1,536 parameters
- Output layer: 16 × 65 = 1,040 parameters
- **Total: ~4,977 parameters**

**New components:**

#### Self-Attention Head
```python
class Head(nn.Module):
    def forward(self, x):
        k = self.key(x)    # what do I have?
        q = self.query(x)  # what am I looking for?
        v = self.value(x)  # what do I communicate?
        
        # Attention scores: which past tokens are relevant?
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = wei.masked_fill(tril == 0, float('-inf'))  # can't see future
        wei = F.softmax(wei, dim=-1)
        
        # Weighted aggregation of values
        out = wei @ v
        return out
```

**Key insight:** Unlike v1 which ignores context, v2 learns which past tokens are relevant for predicting the next one.

**Results:**
- v1 converged to ~2.48 loss
- v2 converges to **~2.41 loss** ✓ (beats baseline!)
- Still runs on CPU (~5k parameters)

### 3. Bigram Language Model v3 (`bigram_v3.py`) 

The real deal - a proper tiny transformer with all the bells and whistles!

**Architecture:**
- Token embeddings: 65 × 64
- Positional embeddings: 32 × 64
- **4 transformer blocks**, each with:
  - Multi-head attention (4 heads)
  - Feed-forward network (expand 4x, then project back)
  - Layer normalization
  - Residual connections
- **Total: ~210k parameters**

**New components:**

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # project back
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

#### Transformer Block
```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # attention + residual
        x = x + self.ffwd(self.ln2(x))  # feed-forward + residual
        return x
```

**Results:**
- Loss: **~1.77** (down from 2.48!)
- Perplexity: ~5.9 (confused between ~6 chars instead of ~12)
- **Still runs on CPU!** (~2-3 min for 5k steps)

**Sample output:**
```
ELIZABETH:
Well.

TRRCALET:
What set-pars;
So luch beast grave are necestinual...
```
It learned character names, dialogue structure, punctuation! 🎉

### 4. Training Loop (`main.py`, `main_v2.py`, `main_v3.py`)

Each training step:
1. **`get_batch()`** - Sample random chunks from Shakespeare
2. **Forward pass** - Predict next characters, compute cross-entropy loss
3. **`optimizer.zero_grad()`** - Clear old gradients
4. **`loss.backward()`** - Backpropagation: compute gradients
5. **`optimizer.step()`** - Update weights to reduce loss

**Hyperparameters:**
- `block_size = 8` - Context window (how many tokens as input)
- `batch_size = 32` - Sequences processed in parallel
- `learning_rate = 1e-3` - Step size for weight updates

### 4. Attention Mechanism (`attention.py`)

Building up to self-attention step by step:

#### Version 1: Naive Loop
```python
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # all tokens up to t
        xbow[b, t] = torch.mean(xprev, dim=0)  # average
```
Simple but slow - O(BT²) explicit loops.

#### Version 2: Matrix Multiplication with `torch.tril`
```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)  # normalize rows
xbow2 = wei @ x  # (T,T) @ (B,T,C) -> (B,T,C)
```
The lower triangular matrix ensures each position only sees past tokens (causal masking).

#### Version 3: Softmax Approach
```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # mask future
wei = F.softmax(wei, dim=1)  # normalize
xbow3 = wei @ x
```
This is how real attention works:
- `-inf` for future positions → `softmax(-inf) = 0` → can't see future
- Currently all past tokens get equal weight
- In real self-attention, weights come from query·key dot products

## Key Concepts

### Why Cross-Entropy Loss?
- Measures how wrong our probability predictions are
- Loss = -log(probability of correct answer)
- Lower is better; 0 = perfect prediction

### What Does the Model Learn?
The embedding table learns transition probabilities:
$$P(\text{next\_char} | \text{current\_char})$$

For example:
- After 'q', 'u' has high probability
- After 'e', many characters possible (space, d, r, s, n...)

### Why Can This Run on CPU?
- v1: Only 4,225 parameters (tiny!)
- v2: Only 4,977 parameters (still tiny!)
- v3: ~210k parameters (still manageable!)

Compare to:
- GPT-2 small: 124 million parameters
- GPT-3: 175 billion parameters

Even v3 with 210k params is **590x smaller** than GPT-2 small!

## Model Comparison

| Model | Parameters | Final Loss | Perplexity | Output Quality |
|-------|-----------|------------|------------|----------------|
| v1 (bigram) | 4,225 | ~2.48 | ~12 | Pure gibberish |
| v2 (attention) | 4,977 | ~2.41 | ~11 | Slightly structured gibberish |
| v3 (transformer) | 209,729 | **~1.77** | **~5.9** | Character names, dialogue, punctuation! |

**The progression:**
- v1 → v2: Added self-attention (+0.07 loss improvement)
- v2 → v3: Added multi-head, feed-forward, layers (**+0.64 loss improvement!**)

## Next Steps

1. ~~Add self-attention~~ ✅ Done in v2!
2. ~~Add positional encoding~~ ✅ Done in v2!
3. ~~Add feed-forward layers~~ ✅ Done in v3!
4. ~~Multi-head attention~~ ✅ Done in v3!
5. ~~Stack multiple layers~~ ✅ Done in v3!
6. **Scale up** - More parameters, longer context, GPU training
7. **Add dropout** - Regularization for better generalization
8. **Train longer** - More iterations for lower loss

## References

- [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
