import os
import json
import argparse
import torch
import torch.nn.functional as F

# Import models
from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using trained character LSTM or Transformer models.")
    
    # Model parameters
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'],
                        help="Model architecture type ('lstm' or 'transformer'). Optional if --generate_all is used.")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the saved model weights (.pt file).")
    parser.add_argument('--seed_text', type=str, default="To be or not to be",
                        help="Seed text to start the generation.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Sampling temperature (lower is more confident/repetitive, higher is more creative/unpredictable).")
    parser.add_argument('--length', type=int, default=500,
                        help="Number of characters to generate.")
    parser.add_argument('--vocab_mappings_path', type=str, default="models/vocab_mappings.json",
                        help="Path to the vocabulary mappings JSON file.")
    parser.add_argument('--meta_path', type=str, default=None,
                        help="Path to model training metadata. If None, resolves automatically based on model choice.")
    
    # Core Requirement 6 Batch Generation
    parser.add_argument('--generate_all', action='store_true',
                        help="Batch generate samples for both models and all temperatures (0.5, 1.0, 1.5) to results/generated_samples.json")
    parser.add_argument('--results_dir', type=str, default="results/",
                        help="Directory to save generated_samples.json")
    
    return parser.parse_args()

def load_vocab(vocab_path):
    """
    Loads character mappings from a JSON file.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary mappings file not found at {vocab_path}. Run prepare_data.py or train.py first.")
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    char_to_int = data["char_to_int"]
    # Convert string keys back to integers for int_to_char mapping
    int_to_char = {int(k): v for k, v in data["int_to_char"].items()}
    vocab = data["vocab"]
    
    return vocab, char_to_int, int_to_char

def load_model_from_weights(model_type, weights_path, meta_path, vocab_size, device):
    """
    Reconstructs the model using stored metadata hyperparams, and loads saved weights.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}.")
        
    # Read model architecture hyperparams from meta JSON if available
    if os.path.exists(meta_path):
        print(f"Loading model architecture metadata from {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        embedding_dim = meta["embedding_dim"]
        hidden_dim = meta["hidden_dim"]
        n_layers = meta["n_layers"]
        n_heads = meta.get("n_heads", 4)
        d_ff = meta.get("d_ff", 256)
        seq_len = meta["seq_len"]
    else:
        # High-quality fallback defaults
        print(f"Metadata file {meta_path} not found. Using default architecture parameters.")
        embedding_dim = 128
        hidden_dim = 256
        n_layers = 2
        n_heads = 4
        d_ff = 256
        seq_len = 100

    if model_type == 'lstm':
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            seq_len=seq_len
        )
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, seq_len

def generate_text(model, model_type, seed_text, length, temperature, char_to_int, int_to_char, seq_len_limit, device):
    """
    Generates text from a seed string.
    Implements temperature-scaled sampling.
    """
    model.eval()
    
    # Process seed text: clean up characters that are not in the vocabulary
    chars = [c for c in seed_text]
    input_ids = [char_to_int[c] for c in chars if c in char_to_int]
    
    # Fallback to spaces or first character in vocab if seed contains no valid characters
    if not input_ids:
        print(f"Warning: Seed text '{seed_text}' contains no valid characters in vocabulary. Initializing with empty spaces.")
        input_ids = [char_to_int.get(' ', 0)]
        chars = [int_to_char[input_ids[0]]]

    with torch.no_grad():
        if model_type == 'lstm':
            # Initialize hidden state
            hidden = model.init_hidden(1, device)
            
            # Warm up LSTM state with the seed prefix (up to the second-to-last character)
            for idx in input_ids[:-1]:
                curr_x = torch.tensor([[idx]], dtype=torch.long, device=device)
                _, hidden = model(curr_x, hidden)
            
            # Autoregressive generation from the last character
            curr_char_idx = input_ids[-1]
            for _ in range(length):
                curr_x = torch.tensor([[curr_char_idx]], dtype=torch.long, device=device)
                outputs, hidden = model(curr_x, hidden)
                # outputs shape: [1, 1, vocab_size]
                logits = outputs[0, 0, :]
                
                # Temperature Scaling
                if temperature <= 0.0:
                    next_char_idx = torch.argmax(logits).item()
                else:
                    # Scale logits and sample
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()
                    
                chars.append(int_to_char[next_char_idx])
                curr_char_idx = next_char_idx
        else:
            # Transformer generation
            for _ in range(length):
                # Slice input sequence history to match the training sequence length limit
                history = input_ids[-seq_len_limit:] if len(input_ids) > seq_len_limit else input_ids
                x = torch.tensor([history], dtype=torch.long, device=device)
                logits = model(x) # shape: [1, seq_len_current, vocab_size]
                last_logit = logits[0, -1, :]
                
                # Temperature Scaling
                if temperature <= 0.0:
                    next_char_idx = torch.argmax(last_logit).item()
                else:
                    probs = F.softmax(last_logit / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()
                    
                chars.append(int_to_char[next_char_idx])
                input_ids.append(next_char_idx)
                
    return "".join(chars)

def run_batch_generation(args, vocab_size, char_to_int, int_to_char, device):
    """
    Core Requirement 6: Automatically generates and saves multiple samples 
    for LSTM and Transformer across temperatures 0.5, 1.0, and 1.5 in results/generated_samples.json
    """
    print("\n==================================================")
    print("Running Batch Generation for comparison...")
    print("==================================================")
    
    results = {
        "lstm": {},
        "transformer": {}
    }
    
    temperatures = [0.5, 1.0, 1.5]
    
    # Seeds for qualitative testing
    seeds = [
        "To be or not to be, that is the ",
        "ROMEO:\nWhat light through yonder window ",
    ]
    
    for m_type in ['lstm', 'transformer']:
        # Establish default paths
        weights_path = os.path.join("models", f"{m_type}_model.pt")
        meta_path = os.path.join("models", f"{m_type}_meta.json")
        
        if not os.path.exists(weights_path):
            print(f"Warning: Trained weights for {m_type.upper()} not found at {weights_path}. Skipping.")
            continue
            
        print(f"\nLoading {m_type.upper()} weights and metadata...")
        model, seq_len_limit = load_model_from_weights(m_type, weights_path, meta_path, vocab_size, device)
        
        for temp in temperatures:
            key = f"temperature_{temp}"
            results[m_type][key] = []
            print(f"Generating 2 samples for {m_type.upper()} at Temperature {temp}...")
            
            for idx, seed in enumerate(seeds):
                generated = generate_text(
                    model=model,
                    model_type=m_type,
                    seed_text=seed,
                    length=200, # Generate modest 200 characters per sample
                    temperature=temp,
                    char_to_int=char_to_int,
                    int_to_char=int_to_char,
                    seq_len_limit=seq_len_limit,
                    device=device
                )
                results[m_type][key].append(generated)
                print(f"--- Sample {idx + 1} (Seed: '{seed.strip()}') ---")
                print(generated)
                print("-" * 40)
                
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "generated_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSuccessfully stored generated samples in JSON format at: {out_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Vocabulary
    vocab, char_to_int, int_to_char = load_vocab(args.vocab_mappings_path)
    vocab_size = len(vocab)
    
    # Check if user requested the batch generation for comparing architectures
    if args.generate_all:
        run_batch_generation(args, vocab_size, char_to_int, int_to_char, device)
        return
        
    # Single run model validation
    if not args.model:
        raise ValueError("Please provide --model (lstm or transformer) for single text generation, or use --generate_all.")
        
    # Resolve default weight and metadata paths if not specified
    weights_path = args.model_path
    if not weights_path:
        weights_path = os.path.join("models", f"{args.model}_model.pt")
        
    meta_path = args.meta_path
    if not meta_path:
        meta_path = os.path.join("models", f"{args.model}_meta.json")
        
    print(f"Loading {args.model.upper()} model...")
    model, seq_len_limit = load_model_from_weights(args.model, weights_path, meta_path, vocab_size, device)
    
    print(f"Generating {args.length} characters with seed: '{args.seed_text}'...")
    print(f"Temperature: {args.temperature} | Hardware: {device}")
    print("=" * 60)
    
    generated_text = generate_text(
        model=model,
        model_type=args.model,
        seed_text=args.seed_text,
        length=args.length,
        temperature=args.temperature,
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        seq_len_limit=seq_len_limit,
        device=device
    )
    
    print(generated_text)
    print("=" * 60)

if __name__ == "__main__":
    main()
