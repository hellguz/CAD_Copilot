# check_max_length.py
import json
import math

# Path to your processed data
data_path = "data/processed/tokenized_floorplans.json"

with open(data_path, 'r') as f:
    data = json.load(f)

max_len = 0
for split in ['train', 'val', 'test']:
    for seq in data[split]:
        if len(seq) > max_len:
            max_len = len(seq)

# Suggest a new value, rounded up to a multiple of 512 for efficiency
suggested_length = math.ceil(max_len / 512) * 512 if max_len > 0 else 2048

print(f"✅ The longest sequence in your dataset has {max_len} tokens.")
print(f"💡 Recommendation: Open 'src/config.py' and set MAX_SEQ_LENGTH to {suggested_length} or higher.")