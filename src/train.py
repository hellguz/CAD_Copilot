import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os

from src import config
from src.model import FloorplanTransformer
from src.dataset import FloorplanDataset, collate_fn

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Create masks
        src_padding_mask = (inputs == config.VOCAB_SIZE).to(device) # Needs correct pad token value
        tgt_mask = FloorplanTransformer.generate_square_subsequent_mask(inputs.size(1), device)

        # Forward pass
        optimizer.zero_grad()
        output = model(inputs, src_mask=tgt_mask, src_padding_mask=src_padding_mask)
        
        # Reshape for loss calculation
        loss = criterion(output.view(-1, config.VOCAB_SIZE), targets.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            src_padding_mask = (inputs == config.VOCAB_SIZE).to(device)
            tgt_mask = FloorplanTransformer.generate_square_subsequent_mask(inputs.size(1), device)
            
            output = model(inputs, src_mask=tgt_mask, src_padding_mask=src_padding_mask)
            loss = criterion(output.view(-1, config.VOCAB_SIZE), targets.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset metadata to update config
    with open(config.PROCESSED_DATA_PATH, 'r') as f:
        meta = json.load(f)['meta']
    config.VOCAB_SIZE = meta['vocab_size']
    
    # Dataloaders
    pad_token_value = meta['pad_token']
    train_dataset = FloorplanDataset(config.PROCESSED_DATA_PATH, 'train', config.MAX_SEQ_LENGTH)
    val_dataset = FloorplanDataset(config.PROCESSED_DATA_PATH, 'val', config.MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_value))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=lambda b: collate_fn(b, pad_token_value))

    # Model, Loss, Optimizer
    model = FloorplanTransformer(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.EMBEDDING_DIM,
        nhead=config.NUM_HEADS,
        d_hid=config.D_FF,
        nlayers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LENGTH
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            print("  -> Saved new best model.")

if __name__ == "__main__":
    main()

