# Dakshina Transliteration Assignment — DA6401

This repository contains the implementation of a character-level sequence-to-sequence (seq2seq) transliteration model using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina). The task is to transliterate Romanized Hindi words into Devanagari script.

---

## Folder Structure

### `Without_Attention/` (Questions 1 to 4)
This folder implements the vanilla seq2seq model using RNN/LSTM/GRU cells without attention.

**Files Included:**
- `ma23m026-a3-da6401.ipynb`: Jupyter notebook implementing and training the vanilla seq2seq model.
- `ma23m026-a3-da6401.py`: Script version of the notebook for batch training.
- `predictions_without_attention.csv`: Final predictions from the vanilla model on the test set.
- `README`: This file.

### `With_Attention/` (Questions 5 to 6)
This folder extends the vanilla model by integrating an attention mechanism into the decoder.

**Files Included:**
- `ma23m026-a3-da6401-attention.ipynb`: Notebook with attention-based encoder-decoder implementation.
- `ma23m026-a3-da6401-attention.py`: Script version of the attention-based model.
- `attention_predictions_new.csv`: Final test predictions using the attention model.
- `README`: This file.

---

## Tasks Completed

| Question | Description |
|----------|-------------|
| Q1       | Built a configurable RNN-based encoder-decoder transliteration model. |
| Q2       | Performed hyperparameter sweep using wandb to identify optimal settings. |
| Q3       | Analyzed results and derived insights from wandb plots and metrics. |
| Q4       | Evaluated the best vanilla model on the test set and analyzed errors. |
| Q5       | Added attention mechanism to the decoder and re-trained the model. |
| Q6       | Visualized attention weights as heatmaps and analyzed focus patterns. |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/viinod9/DA6401_MA23M026_A3.git
cd DA6401_MA23M026_A3
```

### 2. Install dependencies
```bash
pip install wandb 
```
### 3. Run the vanilla model (without attention)

```
cd without_attention
python ma23m026-a3-da6401.py
```

### 4. Run the attention-based model
```
cd with_attention
python ma23m026-a3-da6401-attention.py
Ensure that the dataset is placed at the following path:
/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/
```

### 5. Evaluation
Metric: Accuracy based on exact sequence match between prediction and reference.

Model Comparison: Compared attention-based model with vanilla encoder-decoder.

### 6. Visualizations:

wandb training/validation loss and accuracy plots

3×3 attention heatmaps grid

Sample predictions and qualitative error analysis

### Key Features
Modular code for switching between RNN, LSTM, and GRU.

Flexible setup for embedding size, hidden units, layers, etc.

Integrated wandb sweep for efficient hyperparameter search.

Attention visualization and interpretable decoding behavior.

Separate logging and CSV export of predictions.

### Notes
The test set was only used during final evaluation.

No data leakage between train/dev/test sets.

All experiments and tuning were done using only training and validation data.

Code is cleanly organized, with support for both notebooks and scripts.

### GitHub Submission
GitHub Repository:
```https://github.com/<your-username>/da6401_assignment3
<viinod9>
```

### Contribution Policy
All code is original and adheres to academic integrity guidelines.

Contributions are fairly split and tracked via commit history.

Code is well-commented and contains clear training and evaluation logic.
