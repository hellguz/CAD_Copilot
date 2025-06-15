import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import random

class FloorplanDataset(Dataset):
    def __init__(self, data_path, split='train', max_seq_length=2048):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data[split]
        self.pad_token = data['meta']['pad_token']
        self.max_seq_length = max_seq_length
        
        # NOTE: We no longer filter out long sequences. We will truncate them instead.

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # --- NEW: Truncate long sequences by taking a random slice ---
        # This allows the model to see different parts of long drawings in each epoch.
        if len(sequence) > self.max_seq_length:
            start_idx = random.randint(0, len(sequence) - self.max_seq_length - 1)
            sequence = sequence[start_idx : start_idx + self.max_seq_length]

        # The input is the sequence, and the target is the sequence shifted by one
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)

def collate_fn(batch, pad_token):
    """Custom collate function to pad sequences in a batch."""
    inputs, targets = zip(*batch)
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_token)
    
    return padded_inputs, padded_targets