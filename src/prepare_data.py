import os
import urllib.request
import json
import torch
from torch.utils.data import Dataset

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class CharDataset(Dataset):
    """
    A custom PyTorch Dataset for character-level text sequences.
    Given an encoded list of integers, returns sequence chunks of length seq_len
    along with their target sequences shifted by one position.
    """
    def __init__(self, data_encoded, seq_len):
        self.data = data_encoded
        self.seq_len = seq_len

    def __len__(self):
        # We need seq_len + 1 elements to get a valid (input, target) pair
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y

def download_dataset(dest_path="input/shakespeare.txt"):
    """
    Downloads the TinyShakespeare dataset if it does not already exist.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading TinyShakespeare dataset from {SHAKESPEARE_URL}...")
        try:
            urllib.request.urlretrieve(SHAKESPEARE_URL, dest_path)
            print(f"Successfully downloaded to {dest_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists at {dest_path}")

def load_data(dataset_path="input/shakespeare.txt"):
    """
    Loads raw text dataset.
    """
    if not os.path.exists(dataset_path):
        download_dataset(dataset_path)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def create_vocab_mappings(text, save_dir="models"):
    """
    Creates vocabulary character-to-integer mappings and saves them.
    """
    os.makedirs(save_dir, exist_ok=True)
    vocab = sorted(list(set(text)))
    char_to_int = {char: idx for idx, char in enumerate(vocab)}
    int_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    # Save vocabulary files for reference in text generation
    mapping_path = os.path.join(save_dir, "vocab_mappings.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "char_to_int": char_to_int,
            "int_to_char": {str(k): v for k, v in int_to_char.items()},
            "vocab": vocab
        }, f, indent=4)
        
    return vocab, char_to_int, int_to_char

def encode_text(text, char_to_int):
    """
    Converts text characters to matching integer representations.
    """
    return [char_to_int[char] for char in text]

def get_dataloaders(dataset_path="input/shakespeare.txt", seq_len=100, batch_size=64, val_split=0.1, save_dir="models"):
    """
    Complete pipeline to load, tokenize, split, and package text into PyTorch DataLoaders.
    """
    text = load_data(dataset_path)
    print(f"Dataset total size: {len(text)} characters.")
    
    vocab, char_to_int, int_to_char = create_vocab_mappings(text, save_dir)
    print(f"Vocabulary size: {len(vocab)} unique characters.")
    
    encoded_data = encode_text(text, char_to_int)
    
    # Perform validation split
    split_idx = int(len(encoded_data) * (1 - val_split))
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]
    
    train_dataset = CharDataset(train_data, seq_len)
    val_dataset = CharDataset(val_data, seq_len)
    
    # Pin memory for faster data transfer during training if needed (or keep it default for CPU)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    return train_loader, val_loader, vocab, char_to_int, int_to_char

if __name__ == "__main__":
    # Test script download and setup
    train_loader, val_loader, vocab, _, _ = get_dataloaders(seq_len=100, batch_size=64)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shapes -> Input: {x.shape}, Target: {y.shape}")
