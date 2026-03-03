"""
Bigram Language Model v6 - BPE Tokenization!

The big change: moving from character-level to subword-level (BPE).

Character-level (v1-v5):
  "Hello world" → ['H','e','l','l','o',' ','w','o','r','l','d'] → 11 tokens
  Vocab: 65-174 tokens

BPE / GPT-2 (v6):
  "Hello world" → ['Hello', ' world'] → 2 tokens!
  Vocab: 50,257 tokens

Trade-off:
  - Pro: Each token carries MORE meaning (whole words/subwords)
  - Pro: Fewer tokens = more context in same block_size
  - Con: Larger vocab = bigger embedding table = more parameters
  
Architecture adjustments:
  - n_embd: 64 (reduced from 128 to offset larger embedding table)
  - block_size: 128 (doubled! BPE tokens ≈ 4 chars each, so 128 × 4 = ~512 chars context)
  - n_head: 8, n_layer: 6 (same as v5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Hyperparameters - adjusted for BPE
block_size = 128   # 128 BPE tokens ≈ 512 chars of context (vs 64 chars in v5!)
n_embd = 64        # Reduced from 128 - embedding table is now huge (50k × n_embd)
n_head = 8
n_layer = 6
dropout = 0.1


class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention + feed-forward with residual connections."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    v6: First model that deserves the "GPT" name!
    Uses BPE tokenization just like real GPT-2.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output projection - tied to embedding weights
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    # Test with GPT-2 vocab size
    vocab_size = 50257  # GPT-2 BPE
    model = GPTLanguageModel(vocab_size)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Break down where params come from
    emb_params = vocab_size * n_embd  # shared with lm_head via weight tying
    pos_params = block_size * n_embd
    transformer_params = n_params - emb_params - pos_params
    
    print(f"GPT Language Model v6 (BPE)")
    print(f"=" * 40)
    print(f"Total parameters: {n_params:,}")
    print()
    print(f"Parameter breakdown:")
    print(f"  Token embeddings:  {emb_params:,} ({emb_params/n_params*100:.1f}%) ← shared with output (weight tying)")
    print(f"  Position embeddings: {pos_params:,} ({pos_params/n_params*100:.1f}%)")
    print(f"  Transformer blocks:  {transformer_params:,} ({transformer_params/n_params*100:.1f}%)")
    print()
    print(f"Architecture:")
    print(f"  Vocab size: {vocab_size:,} (GPT-2 BPE)")
    print(f"  n_embd: {n_embd} (reduced from v5's 128 to offset large vocab)")
    print(f"  n_head: {n_head}, n_layer: {n_layer}")
    print(f"  block_size: {block_size} BPE tokens ≈ {block_size * 4} chars of context")
    print()
    
    # Compare
    print(f"Comparison:")
    print(f"  v5 char-level: 1,218,048 params, {64}-char context")
    print(f"  v6 BPE:        {n_params:,} params, ~{block_size * 4}-char context")
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (4, 32))
    y = torch.randint(0, vocab_size, (4, 32))
    logits, loss = model(x, y)
    print(f"\nTest loss: {loss.item():.4f}")
