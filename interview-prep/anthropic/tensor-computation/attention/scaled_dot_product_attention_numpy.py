import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	# Get the dimension of d_k
	d_k = K.shape[-1] 
	attention_scores = Q @ K.T / np.sqrt(d_k)
	if mask is not None:
		attention_scores = np.where(mask==1, attention_scores, -np.inf)
	# Calculate the stable softmax
	scores = attention_scores - np.max(attention_scores, axis=-1, keepdims=True)
	exp_scores = np.exp(scores)
	attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
	# Output
	output = attention_weights @ V
	return (output, attention_weights)