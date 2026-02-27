"""
Bigram Language Model v2 - with self-attention!

Improvements over v1:
1. Positional embeddings - model knows where tokens are in sequence
2. Self-attention head - tokens can attend to relevant past tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Hyperparameters
block_size = 8   # context length
n_embd = 32      # embedding dimension
head_size = 16   # attention head size


class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter, so we register it as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # scale by sqrt(head_size)

        # Mask future positions (causal attention)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax to get attention weights
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class BigramLanguageModelV2(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Token embeddings: each token gets a vector of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings: each position gets a vector of size n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Self-attention head
        self.sa_head = Head(head_size)
        # Final linear layer to get logits
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) - broadcast addition

        # Apply self-attention
        x = self.sa_head(x)  # (B, T, head_size)

        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens (positional embeddings only go up to block_size)
            idx_cond = idx[:, -block_size:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Focus only on last time step
            logits = logits[:, -1, :]  # (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


if __name__ == '__main__':
    # Quick test
    vocab_size = 65
    model = BigramLanguageModelV2(vocab_size)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (4, 8))
    y = torch.randint(0, vocab_size, (4, 8))
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
