import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class FloorplanDataset(Dataset):
    def __init__(self, data_path, split='train', max_seq_length=1024):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data[split]
        self.pad_token = data['meta']['pad_token']
        self.max_seq_length = max_seq_length
        
        # Filter sequences that are too long
        self.sequences = [s for s in self.sequences if len(s) <= self.max_seq_length]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # The sequence is input, and the target is the sequence shifted by one
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)

def collate_fn(batch, pad_token):
    """Custom collate function to pad sequences in a batch."""
    inputs, targets = zip(*batch)
    
    # Pad inputs
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token)
    
    # Pad targets
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_token)
    
    return padded_inputs, padded_targets

