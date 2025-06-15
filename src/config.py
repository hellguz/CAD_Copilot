# # --- Configuration File ---

# # Data Parameters
# PROCESSED_DATA_PATH = "data/processed/tokenized_floorplans.json"
# VOCAB_SIZE = 5003 # This will be updated by process_data.py, placeholder
# MAX_SEQ_LENGTH = 1024 # Max length of a floorplan sequence

# # Model Hyperparameters
# EMBEDDING_DIM = 512 # Dimension of token embeddings
# NUM_LAYERS = 8 # Number of Transformer Decoder layers
# NUM_HEADS = 8 # Number of attention heads
# D_FF = 2048 # Dimension of the feed-forward network
# DROPOUT = 0.1 # Dropout rate

# # Training Parameters
# BATCH_SIZE = 16
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 50
# DEVICE = "cuda" # "cuda" or "cpu"

# # Checkpoint
# MODEL_SAVE_PATH = "models/"

# --- Configuration File for Toy Dataset ---

# Data Parameters
PROCESSED_DATA_PATH = "data/processed/tokenized_floorplans.json"
VOCAB_SIZE = 10004 # Updated to match your actual vocabulary size
MAX_SEQ_LENGTH = 5120 # Increased to fit your long token sequence

# Model Hyperparameters (Drastically reduced for the toy dataset)
EMBEDDING_DIM = 128      # Reduced from 512
NUM_LAYERS = 3         # Reduced from 8
NUM_HEADS = 4          # Reduced from 8
D_FF = 512             # Reduced from 2048 (4 * EMBEDDING_DIM)
DROPOUT = 0.1          # Dropout is less critical here, but can be kept

# Training Parameters
BATCH_SIZE = 1         # CRITICAL: Must be 1 for a single training sample
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20        # Reduced for quick overfitting test
DEVICE = "cuda"        # Use "cuda" if you have a GPU, otherwise "cpu"

# Checkpoint
MODEL_SAVE_PATH = "models/"