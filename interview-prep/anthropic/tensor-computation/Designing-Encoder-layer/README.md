# Transformer Encoder Layer — NumPy

A from-scratch implementation of a **Transformer Encoder Layer** using only NumPy, based on *Attention Is All You Need*.

## Architecture

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual + LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual + LayerNorm
  ↓
Output
```

## Operations

### Multi-Head Attention

```text
Q = XWq
K = XWk
V = XWv

Attention = softmax(QKᵀ / √d_head)V
```

The attention heads are split, computed independently, concatenated, and projected using `Wo`.

### Feed-Forward Network

```text
FFN(X) = ReLU(XW1 + b1)W2 + b2
```

### Layer Normalization

Normalization is performed over the last dimension using population variance:

```text
LayerNorm(x) = γ(x - μ) / √(σ² + ε) + β
```

## Input

```text
X: (batch_size, seq_len, d_model)
```

`d_model` must be divisible by `num_heads`.

## Parameters

```text
W_q, W_k, W_v, W_o
W1, b1, W2, b2
gamma1, beta1
gamma2, beta2
```

## Key Concepts

* Multi-Head Self-Attention
* Scaled Dot-Product Attention
* Stable Softmax
* Residual Connections
* Layer Normalization
* Position-wise Feed-Forward Network
* Batched Tensor Operations

## Reference

Vaswani et al., *Attention Is All You Need*, 2017.
