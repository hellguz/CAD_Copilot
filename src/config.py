# --- Configuration File for 6GB GPU ---

# Data Parameters
PROCESSED_DATA_PATH = "data/processed/tokenized_floorplans.json"
VOCAB_SIZE = 10004
# IMPORTANT: Reduced sequence length to lower memory usage.
MAX_SEQ_LENGTH = 1024

# Model Hyperparameters (Smaller and faster for quick verification)
EMBEDDING_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 1024
DROPOUT = 0.1

# Training Parameters
# ⚠️ CRITICAL: Batch size is the most important parameter to change for memory.
# We start with a small value of 4.
BATCH_SIZE = 6
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
DEVICE = "cuda"

# Checkpoint
MODEL_SAVE_PATH = "models/"