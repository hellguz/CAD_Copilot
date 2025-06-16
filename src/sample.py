import torch
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.model import FloorplanTransformer
from torch.cuda.amp import autocast

def save_plot_as_image(polylines, filename="generated_floorplan.png"):
    """
    Renders the generated polylines into a PNG image file using Matplotlib.
    """
    if not polylines or not polylines[0]:
        print("[No geometry to plot]")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot each polyline with markers for the vertices
    for poly in polylines:
        if len(poly) > 0:
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            ax.plot(x_coords, y_coords, marker='o', markersize=3, linestyle='-')

    # Highlight the very first point of the drawing
    start_point = polylines[0][0]
    ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')

    # Highlight the very last point of the drawing
    end_point = polylines[-1][-1]
    ax.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')

    ax.set_title("Generated Floorplan Output")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    # This is critical to ensure that squares look like squares
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    plt.savefig(filename)
    plt.close(fig) # Free up memory
    print(f"✅ Plot saved successfully to '{filename}'")


def sample_sequence(model, input_tokens, max_len, device, meta):
    """
    Generates a sequence autoregressively starting from a prompt of tokens.
    """
    model.eval()
    eof_token = meta['eof_token']

    generated_sequence = list(input_tokens)
    current_tokens = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        progress_bar = tqdm(range(max_len), desc="Generating")
        for _ in progress_bar:
            # Ensure the input sequence doesn't exceed the model's max length
            seq_len = current_tokens.size(1)
            if seq_len >= config.MAX_SEQ_LENGTH:
                print("\n--- Max sequence length reached, stopping generation. ---")
                break
                
            mask = FloorplanTransformer.generate_square_subsequent_mask(seq_len, device)
            
            # Use autocast for faster inference with mixed precision
            with autocast():
                output = model(current_tokens, src_mask=mask)

            last_token_logits = output[:, -1, :]
            
            # Use top-k sampling for more diverse results
            top_k = 5
            top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k, dim=-1)
            
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_relative_idx = torch.multinomial(probabilities, 1)
            next_token = torch.gather(top_k_indices, -1, next_token_relative_idx).squeeze()

            if next_token.item() == eof_token:
                print("\n--- End of Floorplan token generated ---")
                progress_bar.close()
                break
            
            generated_sequence.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return generated_sequence


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    with open(config.PROCESSED_DATA_PATH, 'r') as f:
        meta = json.load(f)['meta']
    
    model = FloorplanTransformer(
        vocab_size=meta['vocab_size'],
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

    # --- Define the starting prompt ---
    # This prompt contains two separate, unfinished rooms.
    start_drawing = [
        # Room 1 (4m x 4m), a closed rectangle
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
        # Room 2 (3m x 3m), connected to the first room's wall, unfinished
        [[4.0, 2.0], [7.0, 2.0], [7.0, 5.0]]
    ]
    
    # --- Tokenize the prompt ---
    prompt_tokens = []
    for poly in start_drawing:
        for x, y in poly:
            token_x = int(round(x * 100)) + meta['token_offset']
            token_y = int(round(y * 100)) + meta['token_offset']
            prompt_tokens.extend([token_x, token_y])
        # Add an End-of-Polyline token after each shape in the prompt
        prompt_tokens.append(meta['eol_token'])

    print(f"\nStarting generation with a prompt of {len(start_drawing)} polylines...")
    
    # Generate up to 500 new tokens
    all_tokens = sample_sequence(model, prompt_tokens, max_len=500, device=device, meta=meta)

    # --- Decode all tokens for visualization ---
    polylines = []
    current_poly = []
    token_offset = meta['token_offset']
    eol_token = meta['eol_token']

    token_iterator = iter(all_tokens)
    for t1 in token_iterator:
        if t1 == eol_token:
            if current_poly:
                polylines.append(current_poly)
            current_poly = []
            continue
        try:
            t2 = next(token_iterator)
            # Convert back to meters assuming quantization factor was 100
            x = (t1 - token_offset) / 100.0
            y = (t2 - token_offset) / 100.0
            current_poly.append([x, y])
        except StopIteration:
            break
            
    if current_poly:
        polylines.append(current_poly)

    # Call the plotting function
    save_plot_as_image(polylines, filename="generated_floorplan.png")

if __name__ == "__main__":
    # This allows you to run the script directly
    from src import config
    from src.model import FloorplanTransformer
    main()