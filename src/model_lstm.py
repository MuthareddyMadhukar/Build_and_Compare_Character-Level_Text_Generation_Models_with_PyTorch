import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A character-level LSTM model.
    It takes sequence tokens, projects them into a dense embedding space,
    passes them through stacked LSTM layers, and projects the outputs
    back to the vocabulary space to predict the logit of the next character.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 1. Embedding Layer: Converts character integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. LSTM Layer: Processes the sequence of embeddings.
        # batch_first=True makes tensor dimensions [batch, sequence, features]
        # We add dropout between LSTM layers (if n_layers > 1) to regularize
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0.0
        )

        # 3. Fully Connected Layer: Maps LSTM outputs to vocabulary logit space
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass.
        Args:
            x: Input sequence tensor of shape [batch_size, seq_len]
            hidden: Tuple (h_0, c_0) representing hidden and cell states of shape [n_layers, batch_size, hidden_dim]
        Returns:
            out: Logits of shape [batch_size, seq_len, vocab_size]
            hidden: Updated hidden states (h_n, c_n)
        """
        batch_size, seq_len = x.size()
        
        # 1. Embed the input tokens
        # shape: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embeds = self.embedding(x)
        
        # 2. Run LSTM forward
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # 3. Flatten for fully connected projection
        # lstm_out_flat shape: [batch_size * seq_len, hidden_dim]
        lstm_out_flat = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # 4. Project to vocabulary logits
        # out_flat shape: [batch_size * seq_len, vocab_size]
        out_flat = self.fc(lstm_out_flat)
        
        # 5. Reshape back to sequence form [batch_size, seq_len, vocab_size]
        out = out_flat.view(batch_size, seq_len, -1)
        
        return out, hidden

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden state and cell state for the LSTM.
        Both have shape [n_layers, batch_size, hidden_dim].
        """
        if device is None:
            device = next(self.parameters()).device
            
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)

if __name__ == "__main__":
    # Quick shape and execution sanity check
    model = LSTMModel(vocab_size=65, embedding_dim=128, hidden_dim=256, n_layers=2)
    x = torch.randint(0, 65, (4, 100)) # batch_size=4, seq_len=100
    hidden = model.init_hidden(batch_size=4)
    out, new_hidden = model(x, hidden)
    print("Sanity check passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}") # Should be [4, 100, 65]
    print(f"Hidden h shape: {new_hidden[0].shape}, c shape: {new_hidden[1].shape}")
