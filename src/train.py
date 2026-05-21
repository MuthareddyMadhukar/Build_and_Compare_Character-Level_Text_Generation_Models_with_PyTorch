import os
import time
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import local modules
from src.prepare_data import get_dataloaders
from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel

# Load environment variables from .env
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM or Transformer character-level generators on TinyShakespeare.")
    
    # Model choice
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'],
                        help="Which model architecture to train ('lstm' or 'transformer').")
    
    # Hyperparameters (defaults loaded from environment variables if present)
    parser.add_argument('--epochs', type=int, default=int(os.getenv('EPOCHS', 5)),
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', 64)),
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=float(os.getenv('LEARNING_RATE', 0.002)),
                        help="Learning rate for Adam optimizer.")
    parser.add_argument('--seq_len', type=int, default=int(os.getenv('SEQ_LEN', 100)),
                        help="Sequence length for training chunks.")
    
    # Model dimensions
    parser.add_argument('--embedding_dim', type=int, default=int(os.getenv('EMBEDDING_DIM', 128)),
                        help="Embedding dimension size.")
    parser.add_argument('--hidden_dim', type=int, default=int(os.getenv('HIDDEN_DIM', 256)),
                        help="Hidden dimension (LSTM hidden state / Transformer d_model).")
    parser.add_argument('--n_layers', type=int, default=int(os.getenv('N_LAYERS', 2)),
                        help="Number of stacked layers/blocks.")
    parser.add_argument('--n_heads', type=int, default=int(os.getenv('N_HEADS', 4)),
                        help="Number of attention heads (Transformer only).")
    parser.add_argument('--d_ff', type=int, default=int(os.getenv('D_FF', 256)),
                        help="Feed-forward hidden dimension (Transformer only).")
    
    # Other settings
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="Fraction of data reserved for validation.")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="Maximum training batches to process per epoch (useful for quick testing on CPU).")
    parser.add_argument('--dataset_path', type=str, default=os.getenv('DATASET_PATH', 'input/shakespeare.txt'),
                        help="Path to the Shakespeare raw text dataset.")
    parser.add_argument('--model_save_dir', type=str, default=os.getenv('MODEL_SAVE_DIR', 'models/'),
                        help="Directory to save trained model weights and metadata.")
    parser.add_argument('--results_dir', type=str, default=os.getenv('RESULTS_DIR', 'results/'),
                        help="Directory to save results, loss curves, and text samples.")
    parser.add_argument('--device', type=str, default=None,
                        help="Force 'cpu' or 'cuda'/'mps' execution. If None, auto-selects.")
    
    return parser.parse_args()

def generate_quick_sample(model, model_type, char_to_int, int_to_char, device, seq_len_limit, seed="JULIET:", length=100):
    """
    Utility function to generate a quick, creative text sample during training to monitor quality.
    """
    model.eval()
    with torch.no_grad():
        chars = [c for c in seed]
        input_seq = [char_to_int[c] for c in chars if c in char_to_int]
        if not input_seq:
            input_seq = [char_to_int.get(' ', 0)]
            
        if model_type == 'lstm':
            # LSTM generation (recurrent hidden state update)
            hidden = model.init_hidden(1, device)
            # Warm up hidden state with the seed up to the second-to-last character
            for idx in input_seq[:-1]:
                curr_x = torch.tensor([[idx]], dtype=torch.long, device=device)
                _, hidden = model(curr_x, hidden)
            
            # Autoregressive generation
            curr_char_idx = input_seq[-1]
            for _ in range(length):
                curr_x = torch.tensor([[curr_char_idx]], dtype=torch.long, device=device)
                out, hidden = model(curr_x, hidden)
                # out shape: [1, 1, vocab_size]
                logits = out[0, 0, :]
                probs = F.softmax(logits / 0.8, dim=-1) # sample at temperature 0.8
                next_char_idx = torch.multinomial(probs, 1).item()
                chars.append(int_to_char[next_char_idx])
                curr_char_idx = next_char_idx
        else:
            # Transformer generation (autoregressive feeding of historical tokens)
            for _ in range(length):
                # Slice history to match the maximum trained sequence length limit
                x_seq = input_seq[-seq_len_limit:] if len(input_seq) > seq_len_limit else input_seq
                x = torch.tensor([x_seq], dtype=torch.long, device=device)
                logits = model(x) # shape: [1, current_seq_len, vocab_size]
                # Extract logits for the last generated character
                last_logit = logits[0, -1, :]
                probs = F.softmax(last_logit / 0.8, dim=-1) # sample at temperature 0.8
                next_char_idx = torch.multinomial(probs, 1).item()
                chars.append(int_to_char[next_char_idx])
                input_seq.append(next_char_idx)
                
    return "".join(chars)

def plot_loss_curves(results_dir):
    """
    Saves the training loss curves. If both LSTM and Transformer history files are present,
    it plots them together as required by the core specifications.
    """
    os.makedirs(results_dir, exist_ok=True)
    lstm_path = os.path.join(results_dir, "lstm_loss_history.json")
    transformer_path = os.path.join(results_dir, "transformer_loss_history.json")
    
    plt.figure(figsize=(10, 6))
    plotted = False
    
    # Load and plot LSTM history if available
    if os.path.exists(lstm_path):
        with open(lstm_path, "r") as f:
            lstm_data = json.load(f)
        epochs = range(1, len(lstm_data["train_loss"]) + 1)
        plt.plot(epochs, lstm_data["train_loss"], 'o-', color='#1f77b4', linewidth=2.5, label="LSTM Train Loss")
        plt.plot(epochs, lstm_data["val_loss"], 'o--', color='#aec7e8', linewidth=2, label="LSTM Val Loss")
        plotted = True
        
    # Load and plot Transformer history if available
    if os.path.exists(transformer_path):
        with open(transformer_path, "r") as f:
            trans_data = json.load(f)
        epochs = range(1, len(trans_data["train_loss"]) + 1)
        plt.plot(epochs, trans_data["train_loss"], 's-', color='#ff7f0e', linewidth=2.5, label="Transformer Train Loss")
        plt.plot(epochs, trans_data["val_loss"], 's--', color='#ffbb78', linewidth=2, label="Transformer Val Loss")
        plotted = True
        
    if not plotted:
        print("No loss history found to plot.")
        return

    plt.title("Character-Level Text Generation: LSTM vs Transformer Loss Curves", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    save_path = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved/updated at {save_path}")

def main():
    args = parse_args()
    
    # Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # Load Data
    train_loader, val_loader, vocab, char_to_int, int_to_char = get_dataloaders(
        dataset_path=args.dataset_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        save_dir=args.model_save_dir
    )
    vocab_size = len(vocab)
    
    # Instantiate Model
    if args.model == 'lstm':
        print("\n=== Initializing Character LSTM Model ===")
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers
        ).to(device)
    else:
        print("\n=== Initializing Custom Character Transformer Model ===")
        # For our custom Transformer, d_model is set to args.hidden_dim
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.hidden_dim,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            seq_len=args.seq_len
        ).to(device)
        
    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {num_params:,}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Trackers
    train_losses = []
    val_losses = []
    val_perplexities = []
    
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Sequence Length: {args.seq_len} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    if args.max_batches:
        print(f"Note: Training is limited to a maximum of {args.max_batches} batches per epoch.")

    # Main Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        batches_processed = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if args.model == 'lstm':
                # LSTM requires re-initializing the hidden state for each batch
                # since consecutive batches are shuffled and not continuous
                hidden = model.init_hidden(args.batch_size, device)
                outputs, _ = model(inputs, hidden)
            else:
                # Transformer automatically generates and applies the causal mask in forward()
                outputs = model(inputs)
                
            # Flatten outputs and targets for cross-entropy evaluation
            # outputs shape: [batch_size * seq_len, vocab_size]
            # targets shape: [batch_size * seq_len]
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            batches_processed += 1
            
            # Periodically print batch logs
            if (batch_idx + 1) % 100 == 0 or (args.max_batches and (batch_idx + 1) == args.max_batches):
                print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / batches_processed
        train_losses.append(epoch_train_loss)
        
        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                if args.max_batches and val_batch_idx >= int(args.max_batches * args.val_split):
                    # Keep validation subset proportional if training batch size is limited
                    break
                    
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                if args.model == 'lstm':
                    val_hidden = model.init_hidden(args.batch_size, device)
                    val_outputs, _ = model(val_inputs, val_hidden)
                else:
                    val_outputs = model(val_inputs)
                    
                val_loss = criterion(val_outputs.view(-1, vocab_size), val_targets.view(-1))
                val_running_loss += val_loss.item()
                val_batches += 1
                
        epoch_val_loss = val_running_loss / val_batches
        val_losses.append(epoch_val_loss)
        
        # Calculate Perplexity: exp(average validation loss per character)
        epoch_perplexity = math.exp(epoch_val_loss)
        val_perplexities.append(epoch_perplexity)
        
        epoch_time = time.time() - epoch_start_time
        print(f"\n>>> Epoch [{epoch}/{args.epochs}] Summary ({epoch_time:.1f}s) <<<")
        print(f"    Train Loss: {epoch_train_loss:.4f}")
        print(f"    Val Loss:   {epoch_val_loss:.4f}")
        print(f"    Perplexity: {epoch_perplexity:.4f}")
        
        # Generate a live progress sample to observe learning
        progress_sample = generate_quick_sample(
            model, args.model, char_to_int, int_to_char, device, args.seq_len, seed="JULIET:\n", length=80
        )
        print("    [Generated Sample]:")
        print("    " + progress_sample.replace("\n", "\n    "))
        print("-" * 50)
        
    # Training complete, save weights
    model_weights_path = os.path.join(args.model_save_dir, f"{args.model}_model.pt")
    torch.save(model.state_dict(), model_weights_path)
    print(f"\nModel weights saved successfully to {model_weights_path}")
    
    # Save training configuration and model architecture metadata
    meta_path = os.path.join(args.model_save_dir, f"{args.model}_meta.json")
    metadata = {
        "model_type": args.model,
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads if args.model == 'transformer' else None,
        "d_ff": args.d_ff if args.model == 'transformer' else None,
        "seq_len": args.seq_len,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_perplexity": val_perplexities[-1],
        "epochs": args.epochs
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model architecture metadata saved to {meta_path}")
    
    # Save training loss history for subsequent plotting comparison
    history_path = os.path.join(args.results_dir, f"{args.model}_loss_history.json")
    history_data = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "perplexity": val_perplexities
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Plot/Update joint training loss curves
    plot_loss_curves(args.results_dir)
    print("Training process finished.")

if __name__ == "__main__":
    main()
