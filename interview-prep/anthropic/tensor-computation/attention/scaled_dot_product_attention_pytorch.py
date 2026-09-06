import torch


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    attention_scores = Q @ K.transpose(-2, -1) / torch.sqrt(
        torch.tensor(d_k, dtype=Q.dtype)
    )

    if mask is not None:
        attention_scores = torch.where(
            mask == 1,
            attention_scores,
            torch.tensor(float("-inf"), dtype=Q.dtype)
        )

    scores = attention_scores - torch.max(
        attention_scores, dim=-1, keepdim=True
    ).values

    exp_scores = torch.exp(scores)

    attention_weights = exp_scores / torch.sum(
        exp_scores, dim=-1, keepdim=True
    )

    output = attention_weights @ V

    return output, attention_weights