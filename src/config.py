# --- Configuration File for Full Dataset ---

# Data Parameters
PROCESSED_DATA_PATH = "data/processed/tokenized_floorplans.json"
# IMPORTANT: These are placeholders. process_data.py will determine the true
# vocab size. You MUST run check_max_len.py to find the correct max length
# for your full dataset and update MAX_SEQ_LENGTH below.
VOCAB_SIZE = 10004
MAX_SEQ_LENGTH = 5120

# Model Hyperparameters (Medium-sized for a good balance of performance and speed)
EMBEDDING_DIM = 768      # Model's main dimension (similar to GPT-2 small)
NUM_LAYERS = 12        # Number of Transformer blocks (similar to GPT-2 small)
NUM_HEADS = 12         # Number of attention heads (must be a divisor of EMBEDDING_DIM)
D_FF = 3072            # Hidden dimension of the feed-forward network (4 * EMBEDDING_DIM)
DROPOUT = 0.1          # Standard dropout rate for regularization

# Training Parameters
# ⚠️ If you get a "CUDA out of memory" error, reduce BATCH_SIZE first (e.g., to 8, 4, or 2)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50        # Increased for the larger dataset
DEVICE = "cuda"        # Use "cuda" if you have a GPU, otherwise "cpu"

# Checkpoint
MODEL_SAVE_PATH = "models/"