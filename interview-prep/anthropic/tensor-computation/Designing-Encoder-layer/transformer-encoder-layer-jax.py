import jax
import jax.numpy as jnp


def transformer_encoder_layer(
    X: jnp.ndarray,
    weights: dict,
    num_heads: int,
    eps: float = 1e-5
) -> jnp.ndarray:

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

    Q = Q.reshape(B, L, num_heads, d_head)
    K = K.reshape(B, L, num_heads, d_head)
    V = V.reshape(B, L, num_heads, d_head)

    Q = jnp.transpose(Q, (0, 2, 1, 3))
    K = jnp.transpose(K, (0, 2, 1, 3))
    V = jnp.transpose(V, (0, 2, 1, 3))

    # Attention scores
    #
    # Q: (B, H, L, d_head)
    # K: (B, H, L, d_head)
    #
    # scores: (B, H, L, L)

    scores = Q @ jnp.swapaxes(K, -1, -2)
    scores = scores / jnp.sqrt(d_head)

    # Numerically stable softmax
    attention = jax.nn.softmax(scores, axis=-1)

    # Attention × Values
    #
    # (B, H, L, L)
    #       ×
    # (B, H, L, d_head)
    #
    # -> (B, H, L, d_head)

    heads = attention @ V

    # Concatenate heads
    #
    # (B, H, L, d_head)
    #       ↓
    # (B, L, H, d_head)
    #       ↓
    # (B, L, d_model)

    heads = jnp.transpose(heads, (0, 2, 1, 3))
    heads = heads.reshape(B, L, d_model)

    # Output projection
    attn_output = heads @ weights["W_o"]

    # --------------------------------------------------
    # 2. Add & Norm
    # --------------------------------------------------

    Z = X + attn_output

    mean = jnp.mean(Z, axis=-1, keepdims=True)
    var = jnp.mean((Z - mean) ** 2, axis=-1, keepdims=True)

    Z = (Z - mean) / jnp.sqrt(var + eps)

    Z = (
        Z * weights["gamma1"]
        + weights["beta1"]
    )

    # --------------------------------------------------
    # 3. Feed-Forward Network
    # --------------------------------------------------

    hidden = Z @ weights["W1"] + weights["b1"]

    hidden = jax.nn.relu(hidden)

    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # --------------------------------------------------
    # 4. Add & Norm
    # --------------------------------------------------

    output = Z + ffn_output

    mean = jnp.mean(output, axis=-1, keepdims=True)
    var = jnp.mean((output - mean) ** 2, axis=-1, keepdims=True)

    output = (output - mean) / jnp.sqrt(var + eps)

    output = (
        output * weights["gamma2"]
        + weights["beta2"]
    )

    return output