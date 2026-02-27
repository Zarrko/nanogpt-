# NanoGPT - Learning Transformers from Scratch

A minimal implementation for understanding GPT-style language models, following Andrej Karpathy's approach.

## Project Structure

```
nanogpt/
├── main.py              # Training script for bigram v1
├── main_v2.py           # Training script for bigram v2 (with attention)
├── main_v3.py           # Training script for bigram v3 (multi-head + layers)
├── main_v4.py           # Training script for bigram v4 (CPU limits!)
├── bigram.py            # Bigram language model v1
├── bigram_v2.py         # Bigram language model v2 (with self-attention)
├── bigram_v3.py         # Bigram language model v3 (full transformer!)
├── bigram_v4.py         # Bigram language model v4 (scaled up, 1.2M params)
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

# Train the v4 model (pushing CPU limits!) - ~2 hours on CPU
python main_v4.py
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

**Why multiple heads?** A single attention head learns ONE way to look at the past. But language has many patterns:
- One head might learn "what noun came before?"
- Another might learn "what verb is this related to?"
- Another might track punctuation and sentence boundaries

With 4 heads, the model can attend to 4 different things simultaneously!

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        # Create multiple attention heads in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # combine back to original size
    
    def forward(self, x):
        # Each head processes x independently, then concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.proj(out)  # project back
```

**Example:** With `n_embd=64` and `n_head=4`:
- Each head has `head_size = 64/4 = 16` dimensions
- 4 heads × 16 dims = 64 dims (back to original size)

#### Transformer Block

**Why blocks?** A single attention + feed-forward isn't enough. Stacking multiple blocks lets the model build increasingly complex representations:
- Block 1: Learn basic patterns (bigrams, common pairs)
- Block 2: Combine patterns (words, phrases)
- Block 3: Higher-level structure (sentences, dialogue)
- Block 4: Even more abstract patterns

```python
class Block(nn.Module):
    def forward(self, x):
        # Attention: "gather information from other tokens"
        x = x + self.sa(self.ln1(x))    # residual connection!
        
        # Feed-forward: "think about what I gathered"
        x = x + self.ffwd(self.ln2(x))  # residual connection!
        return x
```

**Key ingredients:**

1. **Residual connections** (`x = x + ...`): The input flows directly to output, plus modifications. This helps gradients flow during training and lets the model learn "just add a little adjustment" rather than rebuilding everything from scratch.

2. **Layer normalization** (`ln1`, `ln2`): Normalizes activations to have mean 0, variance 1. Stabilizes training by preventing values from exploding/vanishing.

3. **Feed-forward network**: 
   ```
   input (64) → expand (256) → ReLU → compress (64)
   ```
   After gathering info via attention, this network processes it. The 4x expansion gives the model more capacity to compute complex functions.

**Visual: Data flow through the model**
```
Input: "ROMEO"
    ↓
┌─────────────────────────────┐
│  Token Embedding (65 → 64)  │  "R" → [0.2, -0.5, 0.1, ...]
│  + Position Embedding       │  position 0 → [0.1, 0.3, ...]
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  ├─ Multi-Head Attention    │  "What should I pay attention to?"
│  └─ Feed-Forward            │  "Process what I learned"
└─────────────────────────────┘
    ↓ (repeat 4x)
┌─────────────────────────────┐
│  Transformer Block 4        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Output Layer (64 → 65)     │  → probability for each character
└─────────────────────────────┘
    ↓
Output: ":" (most likely next character)
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

### 4. Bigram Language Model v4 (`bigram_v4.py`) 🔥

Pushing CPU to its limits! Scaled up everything:

**Architecture:**
- `n_embd`: 64 → 128 (larger embeddings)
- `n_head`: 4 → 8 (more attention heads)
- `n_layer`: 4 → 6 (deeper network)
- `block_size`: 32 → 64 (longer context)
- `dropout`: 0.1 (regularization)
- **Total: ~1.2M parameters** (6x larger than v3!)

**Training:**
- 10,000 steps (2x more than v3)
- `batch_size=64`, `learning_rate=3e-4`
- Training time: **~114 minutes on CPU**

**Results:**
- Loss: **~1.56** (down from 1.77!)
- Perplexity: ~4.8 (confused between ~5 chars instead of ~6)

**Sample output:**
```
BUCKINGHAM:
How soul hear unto me love: buy not for their
amber'd accest and so being Clifford, thanks heir
left the Dightnend hath ere you the dangerous,
To his dark unto dead the brought to this bless:

KING RICHARD II:
On marry arms would plack he hath
but. Now to expeals' liestents; thou, I bite
I to blow new down.

DUKE VINCENTIO:
Well, she was to make order that forcew...
```

Real Shakespeare character names! More coherent sentence structure! 🚀

### 5. Training Loop (`main.py`, `main_v2.py`, `main_v3.py`, `main_v4.py`)

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

### 6. Attention Mechanism (`attention.py`)

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
- v4: ~1.2M parameters (pushing the limits!)

Compare to:
- GPT-2 small: 124 million parameters
- GPT-3: 175 billion parameters

Even v4 with 1.2M params is **100x smaller** than GPT-2 small!

## Model Comparison

| Model | Parameters | Final Loss | Perplexity | Training Time | Output Quality |
|-------|-----------|------------|------------|---------------|----------------|
| v1 (bigram) | 4,225 | ~2.48 | ~12 | ~1 min | Pure gibberish |
| v2 (attention) | 4,977 | ~2.41 | ~11 | ~2 min | Slightly structured gibberish |
| v3 (transformer) | 209,729 | ~1.77 | ~5.9 | ~3 min | Character names, dialogue, punctuation |
| v4 (scaled) | **1,212,481** | **~1.56** | **~4.8** | ~114 min | Real names, coherent sentences! |

**The progression:**
- v1 → v2: Added self-attention (+0.07 loss improvement)
- v2 → v3: Added multi-head, feed-forward, layers (+0.64 loss improvement)
- v3 → v4: Scaled up params, depth, context (+0.21 loss improvement)

## Next Steps

1. ~~Add self-attention~~ ✅ Done in v2!
2. ~~Add positional encoding~~ ✅ Done in v2!
3. ~~Add feed-forward layers~~ ✅ Done in v3!
4. ~~Multi-head attention~~ ✅ Done in v3!
5. ~~Stack multiple layers~~ ✅ Done in v3!
6. ~~Scale up~~ ✅ Done in v4! (1.2M params)
7. ~~Add dropout~~ ✅ Done in v4! (0.1)
8. ~~Train longer~~ ✅ Done in v4! (10k steps)
9. **GPU training** - Scale to 10M+ params, 100k+ steps
10. **Larger datasets** - More than Shakespeare
11. **BPE tokenization** - Word-level instead of character-level

## References

- [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
