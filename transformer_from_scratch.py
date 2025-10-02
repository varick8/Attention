import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TokenEmbedding:
    def __init__(self, vocab_size, d_model, weight_tying=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight_tying = weight_tying
        self.weight = np.random.normal(0, 0.02, (vocab_size, d_model))

    def forward(self, tokens):
        return self.weight[tokens]

    def get_output_projection(self):
        if self.weight_tying:
            return self.weight.T
        else:
            return np.random.normal(0, 0.02, (self.d_model, self.vocab_size))


class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class RoPEPositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def apply_rope(self, x, positions):
        batch_size, num_heads, seq_len, d_k = x.shape

        freqs = 1.0 / (10000 ** (np.arange(0, d_k, 2) / d_k))
        freqs = freqs.reshape(1, 1, 1, -1)

        positions = positions.reshape(1, 1, -1, 1)
        angles = positions * freqs

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]

        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals

        result = np.zeros_like(x)
        result[:, :, :, 0::2] = rotated_even
        result[:, :, :, 1::2] = rotated_odd

        return result


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    x_shifted = np.clip(x_shifted, -500, 500)
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-10)
    return exp_x / sum_exp


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, use_rope=False):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.W_q = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_o = np.random.normal(0, 0.02, (d_model, d_model))

        self.attention = ScaledDotProductAttention(self.d_k)

        if self.use_rope:
            self.rope = RoPEPositionalEncoding(self.d_k)

    def forward(self, x, mask=None, positions=None):
        batch_size, seq_len, d_model = x.shape

        Q = np.matmul(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        if self.use_rope and positions is not None:
            Q = self.rope.apply_rope(Q, positions)
            K = self.rope.apply_rope(K, positions)

        attention_output, attention_weights = self.attention.forward(Q, K, V, mask)

        concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = np.matmul(concat, self.W_o)

        return output, attention_weights


class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.normal(0, 0.02, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.normal(0, 0.02, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = gelu(np.matmul(x, self.W1) + self.b1)
        output = np.matmul(hidden, self.W2) + self.b2
        return output


class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, use_rope=False):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, use_rope)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x, mask=None, positions=None):
        x_norm = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attention_output, attention_weights = self.multihead_attention.forward(x_norm, mask, positions)
        x = x + attention_output

        x_norm = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ffn_output = self.ffn.forward(x_norm)
        x = x + ffn_output

        return x, attention_weights


class GPTTransformer:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=1024,
                 use_rope=False, weight_tying=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_rope = use_rope
        self.weight_tying = weight_tying

        self.token_embedding = TokenEmbedding(vocab_size, d_model, weight_tying)

        if not use_rope:
            self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, use_rope))

        self.ln_final_gamma = np.ones(d_model)
        self.ln_final_beta = np.zeros(d_model)

        self.output_projection = self.token_embedding.get_output_projection()

    def create_causal_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        return mask[None, None, :, :]

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape

        x = self.token_embedding.forward(tokens)

        if not self.use_rope:
            x = self.positional_encoding.forward(x)

        causal_mask = self.create_causal_mask(seq_len)
        positions = np.arange(seq_len) if self.use_rope else None

        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block.forward(x, causal_mask, positions)
            attention_weights.append(attn_weights)

        x = layer_norm(x, self.ln_final_gamma, self.ln_final_beta)

        logits = np.matmul(x, self.output_projection)

        last_token_logits = logits[:, -1, :]
        next_token_probs = softmax(last_token_logits)

        return logits, next_token_probs, attention_weights


def visualize_attention(attention_weights, layer_idx=0, head_idx=0, save_path=None):
    """
    Visualize attention distribution for a specific layer and head
    """
    plt.figure(figsize=(10, 8))

    attn_data = attention_weights[layer_idx][0, head_idx, :, :]

    sns.heatmap(attn_data, annot=True, cmap='Blues', fmt='.3f',
                xticklabels=[f'Pos {i}' for i in range(attn_data.shape[1])],
                yticklabels=[f'Pos {i}' for i in range(attn_data.shape[0])])

    plt.title(f'Attention Distribution - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_attention_statistics(attention_weights):
    """
    Plot statistics of attention distributions across layers
    """
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    attention_entropies = []
    attention_maxes = []

    for layer_idx in range(num_layers):
        layer_entropies = []
        layer_maxes = []

        for head_idx in range(num_heads):
            attn = attention_weights[layer_idx][0, head_idx, :, :]

            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()
            max_attention = np.max(attn, axis=-1).mean()

            layer_entropies.append(entropy)
            layer_maxes.append(max_attention)

        attention_entropies.append(np.mean(layer_entropies))
        attention_maxes.append(np.mean(layer_maxes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(num_layers), attention_entropies, 'bo-')
    ax1.set_title('Average Attention Entropy by Layer')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Entropy')
    ax1.grid(True)

    ax2.plot(range(num_layers), attention_maxes, 'ro-')
    ax2.set_title('Average Max Attention by Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Max Attention')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def test_transformer():
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    seq_len = 10
    batch_size = 2

    print("=== Testing Standard Transformer ===")
    model_standard = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                                   use_rope=False, weight_tying=False)

    print("\n=== Testing RoPE Transformer ===")
    model_rope = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                               use_rope=True, weight_tying=False)

    print("\n=== Testing Weight-Tied Transformer ===")
    model_tied = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                               use_rope=False, weight_tying=True)

    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))

    for model_name, model in [("Standard", model_standard), ("RoPE", model_rope), ("Weight-Tied", model_tied)]:
        print(f"\n--- {model_name} Model ---")
        logits, next_token_probs, attention_weights = model.forward(tokens)

        print(f"Input tokens shape: {tokens.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Next token probabilities shape: {next_token_probs.shape}")
        print(f"Number of attention weight matrices: {len(attention_weights)}")
        print(f"Attention weights shape per layer: {attention_weights[0].shape}")
        print(f"Sum of next token probabilities (should be ~1.0): {np.sum(next_token_probs, axis=-1)}")

        if model_name == "Weight-Tied":
            print(f"Weight tying check - Embedding shape: {model.token_embedding.weight.shape}")
            print(f"Weight tying check - Output projection shape: {model.output_projection.shape}")
            print(f"Weight tying enabled: {model.weight_tying}")

    print("\n=== Attention Visualization Demo ===")
    try:
        logits, next_token_probs, attention_weights = model_standard.forward(tokens)

        print("Generating attention heatmap...")
        visualize_attention(attention_weights, layer_idx=0, head_idx=0)

        print("Generating attention statistics...")
        plot_attention_statistics(attention_weights)

    except Exception as e:
        print(f"Visualization failed (might need GUI): {e}")
        print("But attention data is available for analysis!")

    return True

if __name__ == "__main__":
    test_transformer()