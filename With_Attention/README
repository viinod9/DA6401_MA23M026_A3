# Attention-based Seq2Seq Transliteration Model — Dakshina Dataset

This repository provides a PyTorch implementation of an attention-based sequence-to-sequence (Seq2Seq) model for transliterating Romanized Hindi to Devanagari script using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina). The model includes attention mechanisms and integrates with [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and visualization.

---

## Project Structure

```
.
├── ma23m026_a3_da6401_attention.py   # Model, training utilities, attention functions
├── train_attention.py                # Script to train the model using argparse
├── attention_predictions.csv         # Output predictions after test evaluation
├── README.md                         # This file
```

---

## Setup Instructions

### 1. Install Required Libraries

```bash
pip install torch pandas matplotlib wandb
```

### 2. Download Dataset and Font

Make sure the following are available in your working directory or environment:

- Dakshina Dataset:  
  `/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/`

- Noto Sans Devanagari Font (for attention visualization):  
  `/kaggle/input/nato-sans-devnagari/static/NotoSansDevanagari-Regular.ttf`

---

##  Training the Model

###  Run with Default Parameters

```bash
python train_attention.py
```

###  Custom Training Configuration

You can override default parameters from the command line:

```bash
python train_attention.py \
    --embed_dim 64 \
    --hidden_dim 256 \
    --cell_type LSTM \
    --dropout 0.3 \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --teacher_forcing_ratio 0.6 \
    --optimizer adam \
    --output_csv_path attention_predictions.csv \
    --num_examples 50
```

---

## Evaluation and Outputs

-  **Predictions**: Saved in `attention_predictions.csv`
-  **Accuracy**: Exact word-level accuracy is reported and logged to W&B
-  **Visualizations**:
  - Attention heatmaps for 9 random test samples (3x3 grid)
  - Training and validation loss/accuracy plots
  - HTML logs for detailed prediction inspection

---

##  Model Features

- Sequence-to-sequence model with configurable RNN type: `RNN`, `GRU`, or `LSTM`
- Attention mechanism for better alignment between input and output sequences
- Modular utility functions for preprocessing, batching, evaluation, and visualization
- Integrated with Weights & Biases for tracking and sweep management

---

##  W&B Authentication

Before training, log in to your W&B account:

```bash
wandb login
```

Or embed your key directly into the script:

```python
wandb.login(key='YOUR_WANDB_API_KEY')
```

---

##  Output Files

| File Name                 | Description                            |
|--------------------------|----------------------------------------|
| `attention_predictions.csv` | Final predictions on test set          |
| W&B Dashboard            | Logs for loss, accuracy, and attention |
| Attention Heatmaps       | Visual plots of decoder attention      |

---

##  Citation

If you use the Dakshina dataset, please cite the original dataset creators. Details can be found in the [official GitHub repo](https://github.com/google-research-datasets/dakshina).

---

## License

This project is intended for academic and research purposes only.

