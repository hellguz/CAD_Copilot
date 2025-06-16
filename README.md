# Floorplan Copilot: An Experiment in Generative CAD

<p align="center">
  <em>An experimental Transformer model that learns to draw floorplans like an architect, one point at a time.</em>
  <br/><br/>
  <a href="#"><img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen"/></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"/></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch"/></a>
</p>

---

This project is a deep dive into sequence modeling for generative CAD. Inspired by how GitHub Copilot completes code, **Floorplan Copilot** aims to autocomplete architectural drawings. The core idea is to treat a floorplan not as a static image, but as a sequential series of drawing commands (polylines) and train a Transformer model to predict the next point, line, or even an entire room.

The model is trained on thousands of real-world floorplan SVGs, learning the underlying "language" of architectural design—from the high-level structure of walls and rooms down to the placement of windows and fixtures.

### 🚀 Core Features

-   **SVG Data Pipeline:** A robust pipeline to parse, clean, and process complex SVG floorplans into a model-ready format.
-   **Heuristic Sorting:** Intelligently sorts drawing elements by semantic type (walls first, then stairs, windows, etc.) and spatial proximity to mimic a human's drawing workflow. This was a key discovery for improving model performance.
-   **Transformer-Based Model:** A decoder-only Transformer (GPT-like) that learns the sequential patterns in the drawing data.
-   **Generative Sampling:** A sampling script to generate new geometry from a prompt, with visual output saved as a PNG.

### 🧠 How It Works

The project is a complete pipeline from raw data to generative output:

1.  **SVG Conversion (`convert_svg_to_json.py`):** Thousands of SVG files are parsed. It intelligently categorizes over 500 complex layer names into logical groups using keyword heuristics (`WALL`, `STAIR`, `WIN`, etc.) and filters out annotation layers (`TEXT`, `DIMS`). It then performs a conditional spatial sort on simpler layers to create a continuous drawing path.
2.  **Tokenization (`process_data.py`):** The sorted polylines are normalized, quantized (converted to integer coordinates), and serialized into long sequences of tokens, ready for the model.
3.  **Training (`train.py`):** The Transformer model is trained on these sequences to predict the next token (i.e., the next coordinate). It uses Automatic Mixed Precision (`float16`) for faster training.
4.  **Inference (`sample.py`):** A trained model is loaded, given a starting "prompt" (a few lines of a drawing), and autoregressively generates the rest of the drawing, which is then plotted to an image file.

### ⚙️ Project Structure

```
.
├── data/
│   ├── raw/                # JSON files after SVG conversion
│   ├── processed/          # Final tokenized data for the model
│   └── svg/                # Your source SVG files go here
├── models/                 # Saved model checkpoints (.pth)
├── src/
│   ├── config.py           # All hyperparameters and settings
│   ├── dataset.py          # PyTorch Dataset class (handles truncation)
│   ├── model.py            # The Transformer architecture
│   ├── sample.py           # Script to generate drawings
│   └── train.py            # The main training script
├── .gitignore
├── convert_svg_to_json.py  # Script to process your source SVGs
├── find_layer_names.py     # Helper script to discover all layers in your SVGs
├── process_data.py         # Script to tokenize the JSON data
└── README.md
```

### 🛠️ Setup and Installation

This project uses Conda for environment management to avoid dependency issues.

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and Activate Conda Environment**
    > **Note:** We create a new environment to guarantee a clean slate and avoid the dependency conflicts that can happen in the `base` environment.

    ```bash
    # Create a new environment named 'cad_copilot' with Python 3.11
    conda create --name cad_copilot python=3.11

    # Activate the environment
    conda activate cad_copilot
    ```

3.  **Install PyTorch with CUDA**
    Navigate to the [official PyTorch website](https://pytorch.org/get-started/locally/) and get the correct command for your system. It will look like this:
    ```bash
    # Example for CUDA 12.1. Run the command from the PyTorch website!
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Install Other Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### 🚀 Usage: The 4-Step Pipeline

Follow these steps in order to train your own model.

#### **Step 1: Convert Raw SVGs**
Place all your `.svg` files into the `data/svg` folder. Then run the conversion script. It will automatically sort the geometry and save the results in `data/raw`.

```bash
python convert_svg_to_json.py
```
> **Tip:** You can edit the keyword lists at the top of this script to fine-tune how layers are sorted and filtered.

#### **Step 2: Tokenize the Data**
This script takes the JSON files from `data/raw`, processes them into token sequences, and creates the final training file in `data/processed`.

```bash
python process_data.py
```

#### **Step 3: Configure and Train the Model**
Before training, open `src/config.py` and review the settings. The current configuration is optimized for a GPU with ~6-8GB of VRAM.

| Parameter          | Description                                                                                              | Recommended Value          |
| ------------------ | -------------------------------------------------------------------------------------------------------- | -------------------------- |
| `MAX_SEQ_LENGTH`   | The context window for the model. **Crucial for performance.** | `1024` or `2048`           |
| `BATCH_SIZE`       | How many sequences to process at once. **Lower this if you get "Out of Memory" errors.** | `4`, `8`, or `16`          |
| `EMBEDDING_DIM`    | The main dimension of the model. Larger values increase capacity but slow down training.                 | `256` (Fast) or `768` (Full) |
| `NUM_LAYERS`       | The number of Transformer blocks (depth).                                                                | `6` (Fast) or `12` (Full)  |
| `NUM_EPOCHS`       | How many times to loop over the entire dataset.                                                          | `50`-`100`                 |

Once configured, start the training:

```bash
python -m src.train
```
The script will use Automatic Mixed Precision for speed and save the best model to the `models/` directory.

#### **Step 4: Generate a Sample Drawing**
After training is complete, run the sampling script to see what your model has learned!

```bash
python -m src.sample
```
This will generate a `generated_floorplan.png` file in the root directory, showing a drawing completed by your model from a predefined prompt. You can edit the `start_drawing` list in `src/sample.py` to give it different prompts.

