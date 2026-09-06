import jax.numpy as jnp


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    attention_scores = Q @ K.T / jnp.sqrt(d_k)

    if mask is not None:
        attention_scores = jnp.where(
            mask == 1,
            attention_scores,
            -jnp.inf
        )

    scores = attention_scores - jnp.max(
        attention_scores, axis=-1, keepdims=True
    )

    exp_scores = jnp.exp(scores)

    attention_weights = exp_scores / jnp.sum(
        exp_scores, axis=-1, keepdims=True
    )

    output = attention_weights @ V

    return output, attention_weights