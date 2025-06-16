import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os

import config
from model import FloorplanTransformer
from dataset import FloorplanDataset, collate_fn

def train_epoch(model, dataloader, criterion, optimizer, device, pad_token_value):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Create masks
        # CORRECTED: Use the actual pad_token_value for the mask
        src_padding_mask = (inputs == pad_token_value).to(device)
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

def evaluate(model, dataloader, criterion, device, pad_token_value):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # CORRECTED: Use the actual pad_token_value for the mask
            src_padding_mask = (inputs == pad_token_value).to(device)
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
    pad_token_value = meta['pad_token']
    
    # Dataloaders
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


    # Since val_loader can be empty, check before evaluating
    val_loss = float('nan')
    train_loss = float('nan')
    if val_loader:
        val_loss = evaluate(model, val_loader, criterion, device, pad_token_value)
    print(f"Epoch 0/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    best_val_loss = float('inf')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, pad_token_value)
        
        # Since val_loader can be empty, check before evaluating
        val_loss = float('nan')
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device, pad_token_value)
        
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save the model after each epoch for a toy dataset
        # In a real project, you'd save based on best_val_loss
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
        print("  -> Saved model checkpoint.")

if __name__ == "__main__":
    main()