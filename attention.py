"""
Building up to self-attention, step by step.
Start with simple averaging of previous tokens.
"""
import torch

torch.manual_seed(1337)

# Simple example: batch=4, time=8, channels=2
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# We want x[b, t] = mean of x[b, 0], x[b, 1], ..., x[b, t]
# i.e., average of all previous tokens including current

# Version 1: Naive loop approach (slow!)
xbow = torch.zeros((B, T, C))  # "bag of words" - simple average
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t+1, C) - all tokens from 0 to t
        xbow[b, t] = torch.mean(xprev, dim=0)  # average over time

# Version 2: Efficient matrix multiplication with torch.tril
# Create a lower triangular matrix of weights
wei = torch.tril(torch.ones(T, T))
print("Lower triangular matrix (before normalization):")
print(wei)

# Normalize each row to sum to 1 (so we get averages)
wei = wei / wei.sum(dim=1, keepdim=True)
print("\nNormalized weights (each row sums to 1):")
print(wei)

# Matrix multiply: (T, T) @ (B, T, C) -> (B, T, C)
# wei @ x means: for each position, take weighted sum of previous positions
xbow2 = wei @ x  # pytorch broadcasts over batch dimension

print("\nVersion 1 (loop) xbow[0]:")
print(xbow[0])
print("\nVersion 2 (matmul) xbow2[0]:")
print(xbow2[0])
print(f"\nAre they equal? {torch.allclose(xbow, xbow2)}")

# Version 3: Using softmax (how real attention works!)
# Start with zeros, mask future with -inf, then softmax
import torch.nn.functional as F

tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
# Mask future positions with -infinity (softmax(-inf) = 0)
wei3 = wei3.masked_fill(tril == 0, float('-inf'))
print("\n--- Version 3: Softmax approach ---")
print("Before softmax (future masked with -inf):")
print(wei3)

# Softmax normalizes each row to sum to 1
# -inf becomes 0, equal values become equal probabilities
wei3 = F.softmax(wei3, dim=1)
print("\nAfter softmax:")
print(wei3)

xbow3 = wei3 @ x

print("\nVersion 3 (softmax) xbow3[0]:")
print(xbow3[0])
print(f"\nVersions 2 and 3 equal? {torch.allclose(xbow2, xbow3)}")
