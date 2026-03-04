# NanoGPT - Learning Transformers from Scratch

A minimal implementation for understanding GPT-style language models, following Andrej Karpathy's approach.

## Project Structure

```
nanogpt/
├── main.py              # Training script for bigram v1
├── main_v2.py           # Training script for bigram v2 (with attention)
├── main_v3.py           # Training script for bigram v3 (multi-head + layers)
├── main_v4.py           # Training script for bigram v4 (CPU limits!)
├── main_v5.py           # Training script for bigram v5 (optimizations)
├── main_v5_tinystories.py # Training v5 on TinyStories (1.9B tokens!)
├── main_v6_tinystories.py # Training v6 (BPE) on TinyStories
├── main_v7_tinystories.py # Training v7 (10M params!) on TinyStories
├── bigram.py            # Bigram language model v1
├── bigram_v2.py         # Bigram language model v2 (with self-attention)
├── bigram_v3.py         # Bigram language model v3 (full transformer!)
├── bigram_v4.py         # Bigram language model v4 (scaled up, 1.2M params)
├── bigram_v5.py         # Bigram language model v5 (GELU, weight tying, LR schedule)
├── bigram_v6.py         # GPT language model v6 (BPE tokenization!)
├── bigram_v7.py         # GPT language model v7 (10M params!)
├── attention.py         # Step-by-step attention implementation
├── shakespeare_char/    # Character-level Shakespeare dataset (~1M chars)
│   ├── prepare.py       # Downloads and tokenizes data
│   ├── input.txt        # Raw Shakespeare text
│   ├── train.bin        # Tokenized training data
│   ├── val.bin          # Tokenized validation data
│   └── meta.pkl         # Vocab mappings (stoi, itos)
├── tinystories/         # TinyStories dataset (~500M chars)
│   ├── prepare.py       # Downloads from HuggingFace and tokenizes
│   ├── train.bin        # Tokenized training data (generated)
│   ├── val.bin          # Tokenized validation data (generated)
│   └── meta.pkl         # Vocab mappings (generated)
├── tinystories_bpe/     # TinyStories with GPT-2 BPE tokenization
│   ├── prepare.py       # Tokenizes with tiktoken GPT-2 BPE
│   ├── train.bin        # BPE-tokenized training data (generated)
│   ├── val.bin          # BPE-tokenized validation data (generated)
│   └── meta.pkl         # Tokenizer metadata (generated)
├── nanogpt_v7_colab.ipynb # Google Colab notebook (GPU training!)
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

# Train the v5 model (optimizations) - ~2 hours on CPU
python main_v5.py

# Train v5 on TinyStories (where optimizations actually help!) - ~3 hours
pip install datasets  # if needed
python tinystories/prepare.py  # download ~500MB, tokenize
python main_v5_tinystories.py

# Train v6 (BPE) on TinyStories - the real GPT experience!
pip install tiktoken datasets  # if needed
python tinystories_bpe/prepare.py  # tokenize with GPT-2 BPE
python main_v6_tinystories.py

# Train v7 (10M params!) on TinyStories BPE - scaling up!
python main_v7_tinystories.py  # WARNING: ~10-15 hours on CPU!
```

### Google Colab (Free GPU!)

Training v7 on CPU takes 10-15 hours. On Colab's free T4 GPU, it's **~15-30 minutes**.

1. Open [nanogpt_v7_colab.ipynb](nanogpt_v7_colab.ipynb) in Colab
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook → select `nanogpt_v7_colab.ipynb`
2. Enable GPU: Runtime → Change runtime type → **T4 GPU**
3. Run all cells (Ctrl+F9)

The notebook is fully self-contained — it downloads data, defines the model, trains, and generates text, all in one place.

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
- v6: ~3.5M parameters (several hours)
- v7: ~10.6M parameters (10-15 hours! CPU's last stand)

Compare to:
- GPT-2 small: 124 million parameters
- GPT-3: 175 billion parameters

Even v7 with 10.6M params is **12x smaller** than GPT-2 small!

### Data vs Parameters (Overfitting Risk)

**The golden rule:** You want 10-100x more training tokens than parameters.

**Our situation:**
- Shakespeare: ~1M chars → ~900k training tokens
- Model v4/v5: ~1.2M parameters
- Ratio: **~0.75 tokens per parameter** (uh oh!)

**Why we're not completely overfitting:**
1. **Dropout (0.1)** - Randomly zeros out neurons during training, forcing redundancy
2. **Weight decay (1e-2)** - AdamW penalizes large weights, encouraging simpler solutions
3. **Early stopping** - We stop at 10k iters, not training to full convergence
4. **Character-level structure** - Spelling/grammar patterns are learnable even with less data

**How to spot overfitting:**
- Watch the train/val gap: v4 got train ~1.43, val ~1.56 (gap of 0.13 is okay)
- If gap grows to 0.3+ → definitely overfitting
- If val loss stops improving while train loss drops → overfitting

**What to do about it:**
- Increase dropout (0.2 or 0.3)
- Smaller model (~200k-500k params is probably the sweet spot for Shakespeare)
- More data (TinyStories, Wikipedia, books)

For learning purposes, we accept mild overfitting. For production, you'd want more data!

### TinyStories Dataset

Shakespeare is great for learning, but tiny (~1M chars). **TinyStories** is 500x larger!

**What is TinyStories?**
- ~2.5 million short stories written for children
- Simple English vocabulary and grammar
- Created by Microsoft Research specifically for training small language models
- Source: [HuggingFace roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

**Why TinyStories?**
| Dataset | Characters | Train Tokens | Tokens/1.2M Params |
|---------|------------|--------------|-------------------|
| Shakespeare | ~1M | ~900k | 0.75x (underfitting!) |
| TinyStories | ~500M | ~450M | **375x** (plenty!) |

With TinyStories, we can finally scale to larger models without overfitting.

**Setup:**
```bash
# Install datasets library (if not already)
pip install datasets

# Download and prepare TinyStories (~500MB download, may take a few minutes)
python tinystories/prepare.py
```

**Then update your training script** to point to `tinystories/` instead of `shakespeare_char/`.

## Model Comparison

### Shakespeare Results

| Model | Parameters | Final Loss | Perplexity | Training Time | Output Quality |
|-------|-----------|------------|------------|---------------|----------------|
| v1 (bigram) | 4,225 | ~2.48 | ~12 | ~1 min | Pure gibberish |
| v2 (attention) | 4,977 | ~2.41 | ~11 | ~2 min | Slightly structured gibberish |
| v3 (transformer) | 209,729 | ~1.77 | ~5.9 | ~3 min | Character names, dialogue, punctuation |
| v4 (scaled) | **1,212,481** | **~1.56** | **~4.8** | ~114 min | Real names, coherent sentences! |
| v5 (optimized) | 1,204,096 | ~1.86 | ~6.4 | ~166 min | GELU + weight tying + LR schedule |

### TinyStories Results

**Character-level (v5):**

| Model | Steps | Parameters | Train Loss | Val Loss | Train/Val Gap | Training Time |
|-------|-------|-----------|------------|----------|---------------|---------------|
| v5 (10k steps) | 10,000 | 1,218,048 | 1.35 | 1.34 | 0.01 | ~163 min |
| v5 (20k steps) | 20,000 | 1,218,048 | **1.08** | **1.08** | **0.003** | ~213 min |

**BPE (v6):**

| Model | Steps | Parameters | Train Loss | Val Loss | Train/Val Gap | Training Time |
|-------|-------|-----------|------------|----------|---------------|---------------|
| v6 BPE (20k) | 20,000 | 3,523,520 | 3.68 | 3.69 | 0.01 | ~351 min |

> **Note:** v5 and v6 losses are NOT directly comparable! v5 predicts from 174 characters, v6 predicts from 50,257 BPE tokens. Higher vocab = higher loss numbers. The real comparison is output quality.

**The vindication:** v5 achieved **1.08 loss** on TinyStories vs 1.86 on Shakespeare!
- Train/val gap of 0.003 = **zero overfitting** (vs 0.15 gap on Shakespeare)
- With 1563x tokens per parameter, the optimizations finally shine
- Loss was **still dropping** at 20k steps — the model is underfitting (opposite of Shakespeare!)

**The progression (Shakespeare):**
- v1 → v2: Added self-attention (+0.07 loss improvement)
- v2 → v3: Added multi-head, feed-forward, layers (+0.64 loss improvement)
- v3 → v4: Scaled up params, depth, context (+0.21 loss improvement)
- v4 → v5: Added GELU, weight tying, LR schedule (**-0.30 loss regression!**)
- v5 + TinyStories: Same optimizations, 2000x more data → **1.08 loss!** ✓
- v6 + BPE: GPT-2 tokenizer, real words → **3.69 loss** (not comparable, but real English output!)
- v7 + Scale + GPU: 10.6M params, T4 GPU → **2.17 loss** in 83 min! Multi-paragraph stories!

### Why Did v5 Perform WORSE on Shakespeare?

Surprise! The "industry best practices" made things worse on Shakespeare. Here's why:

1. **Weight tying constrains the model** - Forcing input/output to share weights works great with lots of data, but hurts when data is limited. The model has less flexibility.

2. **LR warmup wasted steps** - 500 warmup steps on a 10k run = 5% of training spent ramping up. With tiny data, every step counts.

3. **Shakespeare is too small** - These optimizations were designed for billion-token datasets. At ~900k tokens, they add overhead without benefit.

4. **Overfitting territory** - At ~0.75 tokens/param, we're already overfitting. Regularization tricks for underfitting don't help here.

**The lesson:** "Best practices" from large-scale training don't always transfer to small models/datasets. Always validate on your specific use case!

**The fix:** We tried v5 on TinyStories (1.9B tokens) and it worked! Val loss dropped to **1.08** with essentially zero overfitting. The optimizations were vindicated - they just needed enough data.

**Generated TinyStories sample (v5 char-level, 20k steps):**
```
She had an a laughed Lily and said, "Let's a face!"
So, But the tricket stood the fet down it was a breshing hill.
Ben specoriald a big with and smelled that them hunt to want to be friends
are a goiny. She played and say to her toys Bed.
```
(Learning patterns but producing character-level gibberish. Needs more params!)

**Generated TinyStories sample (v6 BPE, 20k steps):**
```
The little girl said, "Don't get doing. So, Lily. I love you."
She was happy. She said, "Yes, let's go. I'm good, you are
what to be kind." Her mom smiled and she said, "Thank you."

Once upon a time, there was a boy named Tim had a little girl
named Timmy. One day, Timmy who liked to play.
```
**Real words! Real dialogue! Real story structure!** This is the power of BPE — every generated token is an actual word or subword, so there's no more spelling gibberish.
(Learning children's story patterns — names, dialogue, "friends", "played" — but still at 3.5M params, the model is underfitting.)

### v7: Scaling to 10M Parameters

The natural next step: more params! We scaled to ~10.6M to give the transformer more capacity.

#### Architecture: v6 → v7

| | v6 (BPE) | v7 (10M) |
|---|---|---|
| n_embd | 64 | **160** (2.5x wider) |
| n_head | 8 | 8 |
| head_size | 8 | **20** |
| n_layer | 6 | **8** (deeper) |
| block_size | 128 | **256** (2x context) |
| Context window | ~512 chars | **~1024 chars** |
| Embedding params | 3.2M (91%) | **8.0M (76%)** |
| Transformer params | 299k (9%) | **2.5M (23%)** |
| Total params | 3,523,520 | **10,552,800** |

#### The Embedding Tax

With 50k BPE vocab, the embedding table (`vocab_size × n_embd`) eats most of the param budget:

```
v6:  50,257 × 64  = 3.2M  (91% of 3.5M total)
v7:  50,257 × 160 = 8.0M  (76% of 10.6M total)
GPT-2: 50,257 × 768 = 38.6M (31% of 124M total)
```

As models grow, the embedding fraction shrinks — more params go into the transformer blocks where the "thinking" happens. At 10.6M, we're at 76% embedding, which is better than v6's 91% but still embedding-heavy.

**The fix (for future versions):** Decouple embedding dimension from model dimension with a projection layer. Use small embeddings (e.g., 64-dim) and project up to the model dimension (e.g., 256-dim).

#### Training Configuration (v7)

```python
batch_size = 32          # On T4 GPU (16 on CPU)
max_iters = 10_000       # Half of v6's 20k
max_lr = 3e-4            # Same schedule as v5/v6
warmup_steps = 500       # LR warmup then cosine decay
eval_interval = 500
```

**Training:** ~83 minutes on T4 GPU (Google Colab free tier) vs ~10-15 hours on CPU.

**BPE TinyStories Results:**

| Model | Steps | Parameters | Train Loss | Val Loss | Train/Val Gap | Training Time |
|-------|-------|-----------|------------|----------|---------------|---------------|
| v6 BPE (20k) | 20,000 | 3,523,520 | 3.68 | 3.69 | 0.01 | ~351 min (CPU) |
| **v7 BPE (10k)** | **10,000** | **10,552,800** | **2.17** | **2.17** | **0.008** | **~83 min (T4 GPU)** |

**v7 crushed v6:** Loss dropped from 3.69 to 2.17 in half the steps! And loss was still dropping at step 10k — more training would help further.

**Generated TinyStories sample (v7 BPE, 10k steps):**
```
The man was so happy. He said, "It's so nice. I will always remember it."
The man smiled and said, "Thank you for helping me. Now I will always
remember it."

Once upon a time, there was a curious boy called Tom. Tom loved to explore
the world around him. Every day, he would wander around the woods and find
new things to explore.

One day Tom met a rabbit named Sammy. Sammy was very brave and he wanted
to explore a new place.
```

**Compare to v6 (3.5M, 20k steps):**
```
The little girl said, "Don't get doing. So, Lily. I love you."
She was happy. She said, "Yes, let's go. I'm good, you are
what to be kind."
```

v7 generates **multi-paragraph stories** with character names, dialogue, plot arcs, and emotional content. v6 could only manage a few basic sentences.

### v5 Optimizations Explained

#### 1. GELU Activation (vs ReLU)

**ReLU:** `max(0, x)` - Simple cutoff at zero
- Problem: "Dead neurons" - if a neuron outputs negative, gradient = 0, it never recovers

**GELU:** Smooth curve that mostly passes positive values but sometimes lets small negatives through
- Used in GPT-2, GPT-3, BERT
- Smoother gradients → better training
- Think of ReLU as a light switch (on/off), GELU as a dimmer

#### 2. Weight Tying

**The insight:** The input embedding (char → vector) and output projection (vector → char probabilities) are doing opposite jobs. Why not share the same weights?

```python
# Instead of two separate matrices:
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 65 × 128 = 8,320
self.lm_head = nn.Linear(n_embd, vocab_size)                  # 128 × 65 = 8,320

# Tie them together:
self.lm_head.weight = self.token_embedding_table.weight  # Save 8,320 params!
```

**Why it works:**
- If "A" maps to vector [0.5, 0.3, ...], then [0.5, 0.3, ...] should map back to "A"
- Used in GPT-2 and most modern transformers
- Saves parameters AND often improves quality (forces consistent representations)

#### 3. Learning Rate Warmup + Cosine Decay

**Problem:** Starting with a high learning rate can destabilize training early on.

**Solution: Warmup**
- Start with tiny LR (near 0)
- Gradually increase to max LR over first 500 steps
- Lets the model "warm up" before taking big steps

**Problem:** A constant learning rate isn't optimal.

**Solution: Cosine Decay**
- After warmup, follow a cosine curve from max_lr down to min_lr
- Fast initial learning (when loss is high), gentle refinement (when loss is low)

```
LR Schedule:
     ↑
max_lr ┤      ╭──────╮
       │     ╱        ╲
       │    ╱          ╲
       │   ╱            ╲
       │  ╱              ╲
min_lr ┤ ╱                ╲_____
       └──────────────────────────→
         warmup    cosine decay
```

### v6: BPE Tokenization

The biggest conceptual upgrade: moving from character-level to **subword-level** tokenization!

#### What is BPE?

**Character-level** (v1-v5): Every character is a token
```
"Hello world" → ['H','e','l','l','o',' ','w','o','r','l','d'] → 11 tokens
```

**BPE** (Byte Pair Encoding, v6): Common character sequences merge into single tokens
```
"Hello world" → ['Hello', ' world'] → 2 tokens!
```

**How BPE works (like a five-year-old):**
1. Start with individual characters: `['H', 'e', 'l', 'l', 'o']`
2. Find the most common pair → `'l'+'l'` appears a lot → merge into `'ll'`
3. Now find next most common pair → `'He'` → merge into `'He'`
4. Keep merging until you have ~50,000 tokens
5. Common words become single tokens: `"the"`, `"Hello"`, `" and"`
6. Rare words split into subwords: `"tokenization"` → `["token", "ization"]`

GPT-2 uses 50,257 BPE tokens trained on a huge internet corpus.

#### The Architecture Trade-off

| | v5 (char-level) | v6 (BPE) |
|---|---|---|
| Vocab size | 174 | **50,257** |
| Embedding params | 22k | **3.2M** (91% of model!) |
| Transformer params | 1.2M | 299k |
| Total params | 1,218,048 | **3,523,520** |
| n_embd | 128 | 64 (reduced to offset huge vocab) |
| block_size | 64 chars | 128 BPE tokens ≈ **512 chars** |
| Context window | 64 characters | ~512 characters (**8x more!**) |

**Key insight:** With BPE, the embedding table dominates (91% of params!). We reduced `n_embd` from 128→64 to keep things CPU-trainable, but the model still grew 3x.

**Why it's worth it:**
- Each token carries more meaning (whole words vs single letters)
- 8x more context in the same block_size
- Generated text will be **real words** instead of character-level approximations
- This is how GPT-2, GPT-3, ChatGPT actually work!

## Next Steps

1. ~~Add self-attention~~ ✅ Done in v2!
2. ~~Add positional encoding~~ ✅ Done in v2!
3. ~~Add feed-forward layers~~ ✅ Done in v3!
4. ~~Multi-head attention~~ ✅ Done in v3!
5. ~~Stack multiple layers~~ ✅ Done in v3!
6. ~~Scale up~~ ✅ Done in v4! (1.2M params)
7. ~~Add dropout~~ ✅ Done in v4! (0.1)
8. ~~Train longer~~ ✅ Done in v4! (10k steps)
9. ~~GELU activation~~ ✅ Done in v5!
10. ~~Weight tying~~ ✅ Done in v5!
11. ~~LR warmup + cosine decay~~ ✅ Done in v5!
12. ~~Larger datasets~~ ✅ Done! TinyStories (1.9B tokens)
13. ~~Scale to 10M params~~ ✅ Done in v7! (10.6M params, n_embd=160, 8 layers)
14. ~~BPE tokenization~~ ✅ Done in v6! (GPT-2 BPE, 50,257 vocab)
15. ~~GPU training~~ ✅ Done! Google Colab T4 GPU, 83 min for 10k steps
16. **Scale further** - 100M+ params, 100k+ steps
17. **Embedding projection** - Decouple embedding dim from model dim to reduce embedding tax

## References

- [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
 