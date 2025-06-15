import torch
import json

from src import config
from src.model import FloorplanTransformer

def sample_sequence(model, start_sequence, max_len, device, meta):
    """
    Generates a sequence autoregressively.
    """
    model.eval()
    token_offset = meta['token_offset']
    eof_token = meta['eof_token']

    # Convert start sequence (in meters) to tokens
    # Example: start_sequence = [[0.0, 0.0], [5.0, 0.0]]
    input_tokens = []
    for x, y in start_sequence:
        token_x = int(round(x * 100)) + token_offset
        token_y = int(round(y * 100)) + token_offset
        input_tokens.extend([token_x, token_y])

    generated_sequence = input_tokens[:]
    current_tokens = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len - len(input_tokens)):
            seq_len = current_tokens.size(1)
            mask = FloorplanTransformer.generate_square_subsequent_mask(seq_len, device)
            
            output = model(current_tokens, src_mask=mask)
            # Get the logits for the last token in the sequence
            last_token_logits = output[:, -1, :]
            
            # Use top-k sampling for more diverse results
            top_k = 5
            top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k, dim=-1)
            
            # Sample from the top-k distribution
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_relative_idx = torch.multinomial(probabilities, 1)
            next_token = torch.gather(top_k_indices, -1, next_token_relative_idx).squeeze()

            if next_token.item() == eof_token:
                print("--- End of Floorplan token generated ---")
                break
            
            generated_sequence.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return generated_sequence


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    with open(config.PROCESSED_DATA_PATH, 'r') as f:
        meta = json.load(f)['meta']
    config.VOCAB_SIZE = meta['vocab_size']
    
    model = FloorplanTransformer(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.EMBEDDING_DIM,
        nhead=config.NUM_HEADS,
        d_hid=config.D_FF,
        nlayers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LENGTH
    ).to(device)
    
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # --- Generate a sample ---
    # This is your drawing input from Rhino (in meters)
    start_drawing = [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0]] # Start of a rectangle
    
    print(f"\nStarting generation with input: {start_drawing}")
    generated_tokens = sample_sequence(model, start_drawing, max_len=100, device=device, meta=meta)

    # Decode tokens back to coordinates
    points = []
    token_offset = meta['token_offset']
    special_tokens = {v: k for k, v in meta.items() if isinstance(v, int) and 'token' in k}

    token_iterator = iter(generated_tokens)
    try:
        while True:
            t1 = next(token_iterator)
            if t1 in special_tokens:
                points.append(f"<{special_tokens.get(t1).upper()}>")
                continue
            t2 = next(token_iterator)
            if t2 in special_tokens:
                 points.append(f"<{special_tokens.get(t2).upper()}>")
                 continue

            x = (t1 - token_offset) / 100.0
            y = (t2 - token_offset) / 100.0
            points.append((x, y))

    except StopIteration:
        pass # End of sequence
    
    print("\n--- Generated Drawing ---")
    for p in points:
        print(p)


if __name__ == "__main__":
    import os
    main()

