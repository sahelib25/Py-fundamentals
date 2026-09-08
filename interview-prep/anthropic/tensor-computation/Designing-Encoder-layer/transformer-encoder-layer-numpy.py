import numpy as np

def transformer_encoder_layer(
    X: np.ndarray,
    weights: dict,
    num_heads: int,
    eps: float = 1e-5
) -> np.ndarray:

    B, L, d_model = X.shape
    assert d_model % num_heads == 0

    d_head = d_model // num_heads

    # --------------------------------------------------
    # 1. Multi-Head Self-Attention
    # --------------------------------------------------

    Q = X @ weights["W_q"]
    K = X @ weights["W_k"]
    V = X @ weights["W_v"]

    # (B, L, d_model)
    #      ↓
    # (B, num_heads, L, d_head)

    Q = Q.reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)

    # Q @ K^T
    # (B, H, L, d_head) @ (B, H, d_head, L)
    # -> (B, H, L, L)

    scores = Q @ K.transpose(0, 1, 3, 2)
    scores /= np.sqrt(d_head)

    # Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention /= np.sum(attention, axis=-1, keepdims=True)

    # (B, H, L, L) @ (B, H, L, d_head)
    # -> (B, H, L, d_head)

    heads = attention @ V

    # Concatenate heads
    heads = heads.transpose(0, 2, 1, 3)
    heads = heads.reshape(B, L, d_model)

    # Output projection
    attn_output = heads @ weights["W_o"]

    # --------------------------------------------------
    # 2. Add & Norm
    # --------------------------------------------------

    Z = X + attn_output

    mean = np.mean(Z, axis=-1, keepdims=True)
    var = np.mean((Z - mean) ** 2, axis=-1, keepdims=True)

    Z = (Z - mean) / np.sqrt(var + eps)

    Z = (
        Z * weights["gamma1"]
        + weights["beta1"]
    )

    # --------------------------------------------------
    # 3. Feed-Forward Network
    # --------------------------------------------------

    hidden = Z @ weights["W1"] + weights["b1"]

    # ReLU
    hidden = np.maximum(hidden, 0)

    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # --------------------------------------------------
    # 4. Add & Norm
    # --------------------------------------------------

    output = Z + ffn_output

    mean = np.mean(output, axis=-1, keepdims=True)
    var = np.mean((output - mean) ** 2, axis=-1, keepdims=True)

    output = (output - mean) / np.sqrt(var + eps)

    output = (
        output * weights["gamma2"]
        + weights["beta2"]
    )

    return output