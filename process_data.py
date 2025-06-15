#!/usr/bin/env python3
"""
Script to process raw floorplan data into a tokenized format for model training.
"""
import os
import json
import numpy as np
from pathlib import Path

# --- Configuration ---
# This assumes your raw data is exported as JSON files in `data/raw/`
# Each JSON file should contain one floorplan, with a structure like:
# [
#   [[x1, y1], [x2, y2], ...],  // polyline 1
#   [[x1, y1], [x2, y2], ...],  // polyline 2
#   ...
# ]
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Quantization: convert meters to integer centimeters
QUANTIZATION_FACTOR = 100
# Special tokens
PAD_TOKEN = 0  # Padding token
EOL_TOKEN = 1  # End-of-Polyline
EOF_TOKEN = 2  # End-of-Floorplan
TOKEN_OFFSET = 3 # All coordinate tokens will be `coord_val + TOKEN_OFFSET`

def process_single_floorplan(floorplan_data):
    """Normalizes, quantizes, and tokenizes a single floorplan."""
    all_points = [point for polyline in floorplan_data for point in polyline]
    if not all_points:
        return None, None

    # 1. Normalize Coordinate Frame (Shift to origin)
    points_array = np.array(all_points, dtype=float)
    min_coords = points_array.min(axis=0)
    normalized_points = points_array - min_coords

    # Re-assemble polylines with normalized points
    normalized_floorplan = []
    point_idx = 0
    for polyline in floorplan_data:
        num_points = len(polyline)
        normalized_floorplan.append(normalized_points[point_idx:point_idx+num_points].tolist())
        point_idx += num_points

    # 2. Quantize, Tokenize, and create sequential representation
    token_sequence = []
    max_coord_val = 0
    for polyline in normalized_floorplan:
        for x, y in polyline:
            # Quantize and shift to avoid conflict with special tokens
            token_x = int(round(x * QUANTIZATION_FACTOR)) + TOKEN_OFFSET
            token_y = int(round(y * QUANTIZATION_FACTOR)) + TOKEN_OFFSET
            token_sequence.extend([token_x, token_y])
            max_coord_val = max(max_coord_val, token_x, token_y)
        token_sequence.append(EOL_TOKEN) # Add End-of-Polyline token
    
    token_sequence.append(EOF_TOKEN) # Add End-of-Floorplan token
    return token_sequence, max_coord_val

def main():
    """Main processing function."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_files = list(RAW_DATA_DIR.glob("*.json"))
    if not raw_files:
        print(f"Error: No raw data files (.json) found in '{RAW_DATA_DIR}'.")
        print("Please export your Rhino data into this directory first.")
        # Create a dummy file to show what's expected
        dummy_data = [
            [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0], [0.0, 0.0]], # An outer wall
            [[2.0, 0.0], [2.0, 4.0]] # An inner wall
        ]
        with open(RAW_DATA_DIR / "example_floorplan.json", "w") as f:
            json.dump(dummy_data, f, indent=2)
        print(f"Created a dummy data file: '{RAW_DATA_DIR / 'example_floorplan.json'}'")
        return

    all_sequences = []
    overall_max_coord_val = 0
    
    print(f"Processing {len(raw_files)} floorplan files...")
    for file_path in raw_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        sequence, max_val = process_single_floorplan(data)
        if sequence:
            all_sequences.append(sequence)
            overall_max_coord_val = max(overall_max_coord_val, max_val)

    # The vocabulary size is the largest coordinate token value + 1
    vocab_size = overall_max_coord_val + 1
    
    # Split data
    np.random.shuffle(all_sequences)
    n = len(all_sequences)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = all_sequences[:train_size]
    val_data = all_sequences[train_size:train_size + val_size]
    test_data = all_sequences[train_size + val_size:]

    # Save processed data and metadata
    output_data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "meta": {
            "vocab_size": vocab_size,
            "pad_token": PAD_TOKEN,
            "eol_token": EOL_TOKEN,
            "eof_token": EOF_TOKEN,
            "token_offset": TOKEN_OFFSET
        }
    }

    output_file = PROCESSED_DATA_DIR / "tokenized_floorplans.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f)

    print("\nProcessing complete.")
    print(f"Total floorplans: {n}")
    print(f"  - Training set size: {len(train_data)}")
    print(f"  - Validation set size: {len(val_data)}")
    print(f"  - Test set size: {len(test_data)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Processed data saved to: '{output_file}'")

if __name__ == "__main__":
    main()

