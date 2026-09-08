import torch


def transformer_encoder_layer(
    X: torch.Tensor,
    weights: dict,
    num_heads: int,
    eps: float = 1e-5
) -> torch.Tensor:

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
    #       ↓
    # (B, num_heads, L, d_head)

    Q = Q.reshape(B, L, num_heads, d_head).transpose(1, 2)
    K = K.reshape(B, L, num_heads, d_head).transpose(1, 2)
    V = V.reshape(B, L, num_heads, d_head).transpose(1, 2)

    # Q: (B, H, L, d_head)
    # K: (B, H, L, d_head)
    #
    # scores: (B, H, L, L)

    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_head ** 0.5)

    # Numerically stable softmax
    attention = torch.softmax(scores, dim=-1)

    # (B, H, L, L) @ (B, H, L, d_head)
    # -> (B, H, L, d_head)

    heads = attention @ V

    # Concatenate heads
    # (B, H, L, d_head)
    # -> (B, L, H, d_head)
    # -> (B, L, d_model)

    heads = heads.transpose(1, 2)
    heads = heads.reshape(B, L, d_model)

    # Output projection
    attn_output = heads @ weights["W_o"]

    # --------------------------------------------------
    # 2. Add & Norm
    # --------------------------------------------------

    Z = X + attn_output

    mean = Z.mean(dim=-1, keepdim=True)
    var = ((Z - mean) ** 2).mean(dim=-1, keepdim=True)

    Z = (Z - mean) / torch.sqrt(var + eps)

    Z = (
        Z * weights["gamma1"]
        + weights["beta1"]
    )

    # --------------------------------------------------
    # 3. Feed-Forward Network
    # --------------------------------------------------

    hidden = Z @ weights["W1"] + weights["b1"]

    # ReLU
    hidden = torch.relu(hidden)

    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # --------------------------------------------------
    # 4. Add & Norm
    # --------------------------------------------------

    output = Z + ffn_output

    mean = output.mean(dim=-1, keepdim=True)
    var = ((output - mean) ** 2).mean(dim=-1, keepdim=True)

    output = (output - mean) / torch.sqrt(var + eps)

    output = (
        output * weights["gamma2"]
        + weights["beta2"]
    )

    return output