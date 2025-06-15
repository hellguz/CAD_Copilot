# --- Configuration File ---

# Data Parameters
PROCESSED_DATA_PATH = "data/processed/tokenized_floorplans.json"
VOCAB_SIZE = 5003 # This will be updated by process_data.py, placeholder
MAX_SEQ_LENGTH = 1024 # Max length of a floorplan sequence

# Model Hyperparameters
EMBEDDING_DIM = 512 # Dimension of token embeddings
NUM_LAYERS = 8 # Number of Transformer Decoder layers
NUM_HEADS = 8 # Number of attention heads
D_FF = 2048 # Dimension of the feed-forward network
DROPOUT = 0.1 # Dropout rate

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" # "cuda" or "cpu"

# Checkpoint
MODEL_SAVE_PATH = "models/"

