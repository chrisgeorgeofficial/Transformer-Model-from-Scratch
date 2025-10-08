# Build Transformer Model From Scratch 

How to build Transformer Model 🤔 - Resources Included !

## 📁 Repository Contents

### 1. `DifferentClassesinTransformerModel.ipynb`
A learning-focused notebook containing step-by-step implementations of transformer components.

**Contents:**

#### Problem 1: GPT Dataset Preparation
- `batch_loader()` - Creates training batches from raw text data
- Implements sliding window approach for sequence generation

#### Problem 2: Self Attention (Single-Headed)
- `SingleHeadAttention` class
- Implements scaled dot-product attention
- Causal masking for autoregressive generation

#### Problem 3: Multi-Headed Self Attention
- `MultiHeadedSelfAttention` class
- Parallel attention heads
- Concatenation of multiple attention outputs

#### Problem 4: Transformer Block
- Complete transformer block with:
  - Multi-head self-attention
  - Layer normalization
  - Feed-forward network
  - Residual connections (skip connections)

#### Problem 5: GPT Model
- Full GPT implementation
- Token and positional embeddings
- Sequential transformer blocks
- Final layer normalization and projection

#### Problem 6: Text Generation
- `generate()` function
- Auto-regressive sampling using `torch.multinomial()`
- Context window management

### 2. `drake-lyric-generator_TransformerModel.ipynb`
The main notebook containing the complete implementation for generating Drake lyrics.

**Features:**
- Full GPT architecture implementation with proper naming conventions
- Pre-trained model weights loading (`weights.pt`)
- Text generation using the trained model
- Generates 5000 characters of Drake-style lyrics

**Key Components:**
- `GPT` class - Main model architecture
- `TransformerBlock` - Individual transformer block with skip connections
- `MultiHeadedSelfAttention` - Multi-head attention mechanism
- `SingleHeadAttention` - Individual attention head implementation
- `VanillaNeuralNetwork` - Feed-forward network
- `generate()` function - Auto-regressive text generation

**Model Configuration:**
```python
vocab_size = 104
context_length = 128
model_dim = 252
num_blocks = 6
num_heads = 6
```

## 🎓 Learning Resources

This project is based on the following excellent resources:

- **Transformers Explained** - [Code Basics](https://youtu.be/ZhAz268Hdpw?si=tjJOYzC5AurHZF5Z)
- **Neural Networks** - [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=lGqZJfoAP0BBXEHh)
- **Coding a Transformer From Scratch** - [Dev G](https://youtu.be/kNf7VdUAVS8?si=NOa3pI6ztiST8pBS)
- **Interactive Transformer Visualization** - [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- **NeetCode Problems:**
  - [Self Attention](https://neetcode.io/problems/self-attention)
  - [GPT Dataset](https://neetcode.io/problems/gpt-dataset)
  - [Code GPT](https://neetcode.io/problems/code-gpt)
  - [Make GPT Talk Back](https://neetcode.io/problems/make-gpt-talk-back)
- **Drake Lyric Generator Weights:** - [drake-lyric-generator](https://github.com/gptandchill/drake-lyric-generator)
  

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchtyping
```

### Running the Lyric Generator

1. Clone the repository:
```bash
git clone https://github.com/gptandchill/drake-lyric-generator
cd drake-lyric-generator
```

2. Open `drake-lyric-generator_TransformerModel.ipynb` in Google Colab or Jupyter Notebook

3. Run all cells to generate Drake-style lyrics

### Learning the Components

Open `DifferentClassesinTransformerModel.ipynb` to understand each component step-by-step:
1. Start with data loading
2. Implement single-head attention
3. Build multi-head attention
4. Construct transformer blocks
5. Assemble the complete GPT model
6. Generate text

## 🏗️ Architecture

```
GPT Model
├── Token Embedding (vocab_size → model_dim)
├── Positional Embedding (context_length → model_dim)
├── Transformer Blocks (×6)
│   ├── Multi-Head Self-Attention (6 heads)
│   │   ├── Query, Key, Value projections
│   │   ├── Scaled dot-product attention
│   │   └── Causal masking
│   ├── Layer Normalization
│   ├── Feed-Forward Network (model_dim → 4×model_dim → model_dim)
│   └── Residual Connections
├── Final Layer Normalization
└── Vocabulary Projection (model_dim → vocab_size)
```

## 📊 Key Features

- **Character-level generation**: Works at character granularity for fine control
- **Causal attention**: Ensures autoregressive property
- **Pre-trained weights**: Includes trained model for immediate use
- **GPU support**: Automatically uses CUDA if available
- **Dropout regularization**: 0.2 dropout for better generalization

## 🎯 Sample Output

The model generates lyrics like:
```
[Verse 1: Drake]
Yeah, yeah
Drake, don't fuck what you do what you do
And what you do, yeah
Let you know, what's up on your lion
Girl, I don't cause
I ain't not her, I'll posogrouthing, dancing with you
What it is for it, I make it, this just not tild go
...
```

## 📝 Technical Details

### Model Specifications
- **Context Length**: 128 tokens
- **Model Dimension**: 252
- **Number of Layers**: 6
- **Attention Heads**: 6 per layer
- **Vocabulary Size**: 104 characters
- **Feed-Forward Dimension**: 1008 (4× model_dim)
- **Dropout Rate**: 0.2

### Architecture Components

#### 1. Token & Positional Embeddings
```python
self.token_embedding = nn.Embedding(vocab_size, model_dim)
self.pos_embedding = nn.Embedding(context_length, model_dim)
```

#### 2. Transformer Block
Each block contains:
- **Multi-Head Self-Attention**: 6 parallel attention heads
- **Layer Normalization**: Applied before attention and feed-forward
- **Feed-Forward Network**: Two linear layers with ReLU activation
- **Residual Connections**: Skip connections around each sub-layer

#### 3. Attention Mechanism
- **Scaled Dot-Product Attention**: `softmax(QK^T / √d_k)V`
- **Causal Masking**: Prevents attending to future tokens
- **Head Dimension**: 42 (model_dim / num_heads)

## 🔧 Implementation Details

### Key Differences Between Notebooks

**Drake Lyric Generator (Working Model):**
- Uses naming convention: `token_embedding`, `transformer_blocks`, `mhsa`
- Includes output projection and dropout in attention
- Properly moves tensors to device (`.to(device)`)
- Simplified generation function

**Different Classes (Learning Version):**
- Uses naming convention: `token_embeddings`, `blocks`, `attention`
- Step-by-step component building
- Educational focus with separate problem solutions
- Manual seed state management in generation

### Loading Pre-trained Weights

The model expects weights with these key names:
```python
# Embeddings
token_embedding.weight
pos_embedding.weight

# Transformer blocks
transformer_blocks.{i}.mhsa.attention_heads.{j}.{query/key/value}_layer.weight
transformer_blocks.{i}.mhsa.compute.{weight/bias}
transformer_blocks.{i}.vanilla_nn.first_linear_layer.{weight/bias}
transformer_blocks.{i}.vanilla_nn.second_linear_layer.{weight/bias}
transformer_blocks.{i}.layer_norm_{one/two}.{weight/bias}

# Final layers
layer_norm_three.{weight/bias}
vocab_projection.{weight/bias}
```

## 💡 How It Works

### 1. Input Processing
```python
# Input: "Hello" → [H, e, l, l, o]
# Tokenization: [36, 62, 69, 69, 72]
# Embedding: Each token → 252-dim vector
# Position: Add positional encoding
```

### 2. Transformer Processing
```python
for each block:
    # Self-Attention
    x = x + MultiHeadAttention(LayerNorm(x))
    
    # Feed-Forward
    x = x + FeedForward(LayerNorm(x))
```

### 3. Text Generation
```python
# Auto-regressive generation
for i in range(num_new_chars):
    logits = model(context)
    probs = softmax(logits[-1])
    next_token = sample(probs)
    context = append(context, next_token)
```

## 🎮 Usage Examples

### Basic Generation
```python
# Initialize model
model = GPT(vocab_size=104, context_length=128, 
            model_dim=252, num_blocks=6, num_heads=6)

# Load pre-trained weights
model.load_state_dict(torch.load('weights.pt'))
model.eval()

# Generate lyrics
context = torch.zeros(1, 1, dtype=torch.int64)
lyrics = generate(model, new_chars=5000, context=context, 
                 context_length=128, int_to_char=char_map)
print(lyrics)
```

### Custom Starting Prompt
```python
# Convert prompt to token indices
prompt = "Yeah, I'm in the"
context = torch.tensor([[char_to_int[c] for c in prompt]])

# Generate continuation
lyrics = generate(model, new_chars=1000, context=context,
                 context_length=128, int_to_char=int_to_char)
```

## 🐛 Common Issues & Solutions

### Issue 1: RuntimeError - State Dict Mismatch
**Problem**: Layer names don't match between model and weights.

**Solution**: Ensure your model uses the correct naming convention:
- `token_embedding` (not `token_embeddings`)
- `transformer_blocks` (not `blocks`)
- `mhsa` (not `attention`)

### Issue 2: CUDA Out of Memory
**Problem**: GPU memory exhausted during generation.

**Solution**:
```python
# Reduce batch size or context length
# Or use CPU
device = torch.device("cpu")
```

### Issue 3: Repetitive Output
**Problem**: Model generates repetitive text.

**Solution**:
- Increase temperature in sampling
- Use top-k or nucleus sampling
- Ensure dropout is active during training

## 📚 Understanding the Code

### Attention Mechanism
```python
# Q, K, V projections
Q = self.query(x)  # (batch, seq_len, head_dim)
K = self.key(x)
V = self.value(x)

# Attention scores
scores = (Q @ K.T) / sqrt(head_dim)

# Causal mask (prevents looking ahead)
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -inf)

# Weighted sum
attention = softmax(scores) @ V
```

### Feed-Forward Network
```python
# Expansion and contraction
x = Linear(model_dim → 4*model_dim)  # Expand
x = ReLU(x)                           # Non-linearity
x = Linear(4*model_dim → model_dim)  # Contract
x = Dropout(x)                        # Regularization
```

## 🚀 Future Improvements

- [ ] Add temperature control for generation
- [ ] Implement top-k and nucleus sampling
- [ ] Add beam search for better quality
- [ ] Train on larger corpus
- [ ] Implement word-level tokenization
- [ ] Add model checkpointing during training
- [ ] Create interactive web interface
- [ ] Add lyrics evaluation metrics

## 📖 Learning Path

If you're new to transformers, follow this order:

1. **Start with `DifferentClassesinTransformerModel.ipynb`**
   - Understand each component individually
   - Run each cell and observe outputs
   - Experiment with different parameters

2. **Study the attention mechanism**
   - Visualize attention weights
   - Understand Q, K, V projections
   - Learn about causal masking

3. **Build the complete model**
   - Combine all components
   - Understand data flow
   - Debug shape mismatches

4. **Generate text**
   - Load pre-trained weights
   - Run `drake-lyric-generator_TransformerModel.ipynb`
   - Experiment with generation parameters

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution
- Better training data preprocessing
- Advanced sampling techniques
- Model architecture improvements
- Documentation enhancements
- Bug fixes

## 📜 License

This project is for educational purposes. Drake lyrics are property of their respective copyright holders.

---

⭐ **If you found this project helpful, please star the repository!**

💬 **Questions? Open an issue and let's discuss!**

🎵 **Happy Learning & Happy lyric generating!**
