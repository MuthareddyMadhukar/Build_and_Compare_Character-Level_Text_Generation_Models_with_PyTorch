import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of characters in the sequence.
    Formula:
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Using exp and log for mathematical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (non-trainable tensor saved in state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Embedded input tensor of shape [batch_size, seq_len, d_model]
        """
        # Add the pre-computed positional encoding up to the current sequence length
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    Splits queries, keys, and values into n_heads, calculates scaled dot-product attention
    in parallel for each head, concatenates the attention heads, and projects the result.
    Includes support for causal masks to prevent attending to future characters.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask of shape [batch_size, 1, seq_len, seq_len] or [1, 1, seq_len, seq_len]
                  which is True/1 for positions to keep and False/0 for positions to mask.
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Project inputs to Q, K, V and split into heads
        # Shape transition: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention: Q * K^T / sqrt(d_k)
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Apply causal mask (if provided)
        if mask is not None:
            # We replace masked positions (False/0) with a very large negative value before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Softmax and Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Multiply attention weights by V
        # context shape: [batch_size, n_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, v)
        
        # 6. Concatenate attention heads and project
        # context shape transition: [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        out = self.out_proj(context)
        return out

class FeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network.
    Applies two linear layers with a non-linear activation (GELU) in between.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        """
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    A single Transformer Encoder Block.
    Combines Multi-Head Attention and Position-Wise Feed-Forward.
    Uses Pre-Layer Normalization (Pre-LN) for superior training stability.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Causal attention mask
        """
        # Pre-LN: Norm -> Attention -> Residual Add
        x = x + self.attn(self.norm1(x), mask=mask)
        # Pre-LN: Norm -> Feed-Forward -> Residual Add
        x = x + self.ff(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    """
    A custom character-level Transformer model.
    Stacks token embeddings, positional encodings, and several Transformer blocks.
    A causal mask is automatically generated and applied at each forward pass
    to support autoregressive generation.
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Token Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encodings (set max_len larger than seq_len for safety during generation)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 1000, dropout=dropout)
        
        # Stacks of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Project representation back to character vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass.
        Args:
            x: Input sequence integer IDs of shape [batch_size, seq_len]
            mask: Optional custom mask. If None, an autoregressive causal mask is generated.
        Returns:
            logits: Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        
        # Create a causal upper triangular mask if not provided
        if mask is None:
            # tril creates a lower-triangular matrix of 1s (True for valid, False for masked futures)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            # Reshape to [1, 1, seq_len, seq_len] for proper broadcasting across heads
            mask = mask.unsqueeze(0).unsqueeze(1)
            
        # 1. Project tokens into embeddings and scale them by sqrt(d_model) (original attention paper style)
        embeds = self.embedding(x) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding
        out = self.pos_encoding(embeds)
        
        # 3. Pass through all stacked Transformer blocks
        for block in self.blocks:
            out = block(out, mask=mask)
            
        # 4. Final normalization
        out = self.norm(out)
        
        # 5. Project back to vocabulary logit space
        logits = self.fc(out) # shape: [batch_size, seq_len, vocab_size]
        
        return logits

if __name__ == "__main__":
    # Test model shape and forward execution
    device = torch.device("cpu")
    model = TransformerModel(
        vocab_size=65, 
        d_model=128, 
        n_heads=4, 
        d_ff=256, 
        n_layers=2, 
        seq_len=100, 
        dropout=0.1
    ).to(device)
    
    x = torch.randint(0, 65, (4, 100), device=device) # batch_size=4, seq_len=100
    logits = model(x)
    
    print("Sanity check passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}") # Should be [4, 100, 65]
