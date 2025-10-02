# Transformer From Scratch - NumPy Implementation

This repository contains a complete implementation of a GPT-style Transformer model built from scratch using only NumPy, as part of an individual assignment to understand the core architecture of Transformers.

## Developer

| Nama | NIM |
|------|-----|
| Varick Zahir Sarjiman | 22/496418/TK/54384 |

## Features

### Core Architecture
- **Token Embedding**: Converts input tokens to dense vector representations
- **Positional Encoding**: Multiple options available:
  - Sinusoidal positional encoding (standard)
  - RoPE (Rotary Positional Encoding) for better relative position handling
- **Scaled Dot-Product Attention**: Core attention mechanism with softmax normalization
- **Multi-Head Attention**: Parallel attention heads with Q, K, V projections
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Residual Connections + Layer Normalization**: Pre-norm architecture for stable training
- **Causal Masking**: Prevents access to future tokens (autoregressive generation)
- **Output Layer**: Projects to vocabulary size with softmax distribution

### Bonus Features
- **🎨 Attention Visualization**: Interactive heatmaps and statistical analysis
- **🔗 Weight Tying**: Shared weights between embedding and output layers
- **🌀 RoPE Encoding**: Advanced rotary positional encoding alternative

## Dependencies

- **NumPy**: Core mathematical operations
- **Matplotlib**: For attention visualization plots
- **Seaborn**: Enhanced heatmap visualizations

```bash
pip install numpy matplotlib seaborn
```

## How to Run

1. Clone or download the repository
2. Install dependencies: `pip install numpy matplotlib seaborn`
3. Run the test script:

```bash
python transformer_from_scratch.py
```

## File Structure

- `transformer_from_scratch.py`: Complete implementation with all components and test function
- `README.md`: This documentation file