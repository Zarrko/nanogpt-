"""
GPT Language Model v7 - Scaled to 10M parameters!

Scaling up from v6 (3.5M) to ~10M parameters.

Key changes from v6:
  - n_embd: 64 → 160 (2.5x wider)
  - n_layer: 6 → 8 (deeper)
  - block_size: 128 → 256 (2x context, ~1024 chars)
  - Same: n_head=8, dropout=0.1, GPT-2 BPE (50,257 vocab)

The challenge: With BPE vocab of 50k, the embedding table is HUGE.
At n_embd=160: embedding = 50,257 × 160 = 8M params (most of the model!)
This is a known issue - GPT-2 Small (124M params) has 38M in embeddings (31%).
At our scale, embeddings dominate even more.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Hyperparameters - scaled up for 10M params
block_size = 256   # 256 BPE tokens ≈ 1024 chars of context!
n_embd = 160       # Wider embeddings
n_head = 8         # 8 heads × 20 dims each
n_layer = 8        # Deeper transformer
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
    v7: 10M parameter GPT with BPE tokenization.
    Scaled up from v6 (3.5M) with wider embeddings and deeper transformer.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight  # weight tying
        
        # GPT-2 style initialization (std=0.02)
        # Without this, default PyTorch init (std=1.0) causes logits with std≈√n_embd≈12.6
        # That makes softmax extremely peaked on wrong tokens → initial loss ~100 instead of ~10.8
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    vocab_size = 50257
    model = GPTLanguageModel(vocab_size)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    emb_params = vocab_size * n_embd
    pos_params = block_size * n_embd
    transformer_params = n_params - emb_params - pos_params
    
    print(f"GPT Language Model v7 (10M params)")
    print(f"=" * 40)
    print(f"Total parameters: {n_params:,}")
    print()
    print(f"Parameter breakdown:")
    print(f"  Token embeddings:    {emb_params:,} ({emb_params/n_params*100:.1f}%) ← weight tied")
    print(f"  Position embeddings: {pos_params:,} ({pos_params/n_params*100:.1f}%)")
    print(f"  Transformer blocks:  {transformer_params:,} ({transformer_params/n_params*100:.1f}%)")
    print()
    print(f"Architecture:")
    print(f"  n_embd: {n_embd}, n_head: {n_head}, n_layer: {n_layer}")
    print(f"  head_size: {n_embd // n_head}")
    print(f"  block_size: {block_size} BPE tokens ≈ {block_size * 4} chars")
    print()
    print(f"Scale comparison:")
    print(f"  v5 (char):  1.2M params, 64-char context")
    print(f"  v6 (BPE):   3.5M params, ~512-char context")
    print(f"  v7 (10M):   {n_params/1e6:.1f}M params, ~{block_size * 4}-char context")
    
    x = torch.randint(0, vocab_size, (2, 32))
    y = torch.randint(0, vocab_size, (2, 32))
    logits, loss = model(x, y)
    print(f"\nTest loss: {loss.item():.4f}")
