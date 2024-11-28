# Transformer: Attention is All You Need - PyTorch Implementation

## 🌟 Project Overview
This repository contains a detailed PyTorch implementation of the Transformer architecture from the groundbreaking research paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The implementation provides a modular and flexible approach to building sequence-to-sequence models using self-attention mechanisms.

## 📜 Research Paper Reference
**Title:** Attention Is All You Need
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**Published:** June 2017, NeurIPS

## 🔍 Key Components Breakdown

### 1. Self-Attention Mechanism (`SelfAttention` Class)
The core innovation of the Transformer architecture. Key features:
- Multi-head attention implementation
- Computes attention across different representation subspaces
- Allows parallel processing of sequence elements

#### Key Methods:
- `__init__()`: Initializes linear transformations for values, keys, and queries
- `forward()`: Calculates attention weights and applies them to input

### 2. Transformer Block (`TransformerBlock` Class)
Combines self-attention with feed-forward neural network:
- Applies layer normalization
- Includes dropout for regularization
- Residual connections for improved gradient flow

### 3. Encoder (`Encoder` Class)
Responsible for processing input sequences:
- Word embeddings
- Positional embeddings
- Multiple transformer block layers
- Supports configurable number of layers and embedding sizes

### 4. Decoder (`Decoder` Class)
Generates output sequences:
- Masked self-attention to prevent looking ahead
- Converts embeddings to vocabulary space
- Supports multiple decoder layers

### 5. Main Transformer (`Transformer` Class)
Orchestrates the entire translation process:
- Combines encoder and decoder
- Creates source and target masks
- Handles device (CUDA) configuration

## 🛠️ Hyperparameters

### Configurable Parameters
- `src_vocab_size`: Source vocabulary size
- `trg_vocab_size`: Target vocabulary size
- `embed_size`: Embedding dimension (default: 256)
- `num_layers`: Number of transformer blocks (default: 6)
- `heads`: Number of attention heads (default: 8)
- `forward_expansion`: Feed-forward network expansion factor (default: 4)
- `dropout`: Regularization dropout rate (default: 0)
- `max_length`: Maximum sequence length (default: 100)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7.0+
- CUDA (recommended for GPU acceleration)

### Clone Repository
```bash
git clone https://github.com/yourusername/transformer-implementation.git
cd transformer-implementation
```

### Install Dependencies
```bash
pip install torch
```

## 💻 Usage Example

```python
# Create Transformer model
model = Transformer(
    src_vocab_size=10000,
    trg_vocab_size=10000,
    src_pad_idx=0,
    trg_vocab_idx=0,
    embed_size=256,
    num_layers=6,
    heads=8,
    dropout=0.1
)

# Prepare input sequences
src = torch.randint(0, 10000, (32, 50))  # Batch size 32, sequence length 50
trg = torch.randint(0, 10000, (32, 50))  # Target sequence

# Forward pass
output = model(src, trg)
```

## 🧠 Key Innovations

### 1. Self-Attention Mechanism
- Replaces traditional recurrence and convolution
- Allows direct modeling of dependencies regardless of sequence distance
- Enables parallel computation of sequence elements

### 2. Multi-Head Attention
- Jointly attends to information from different representation subspaces
- Captures various types of relationships within the input

### 3. Positional Encoding
- Injects information about token positions
- Crucial for sequence modeling without recurrence

## 🔬 Technical Details

### Attention Calculation
- Uses `torch.einsum` for efficient attention weight computation
- Supports flexible masking mechanisms
- Handles variable-length sequences

### Masking Strategies
- Source mask (`make_src_mask`): Handles padding in input sequences
- Target mask (`make_trg_mask`): Prevents attending to future tokens during decoding

## 🚧 Current Limitations
- Fixed maximum sequence length
- Computationally expensive for very long sequences
- Potential numerical instability in attention calculations

## 🔮 Potential Improvements
- Implement adaptive attention span
- Add more advanced positional encodings
- Explore sparse attention mechanisms
- Enhance numerical stability

## 📊 Performance Considerations
- GPU acceleration support
- Configurable dropout for regularization
- Modular design for easy experimentation

## 🤝 Contributing
Contributions are welcome! Please submit pull requests or open issues.

## 📝 License
MIT License

## 📚 Citation
```bibtex
@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

## 🌈 Acknowledgments
Inspired by the groundbreaking work of Vaswani et al.
