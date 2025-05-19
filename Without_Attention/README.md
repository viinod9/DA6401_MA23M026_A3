
# Vanilla Seq2Seq Transliteration Model — Dakshina Dataset

This repository contains an implementation of a vanilla sequence-to-sequence (Seq2Seq) model using PyTorch to transliterate Romanized Hindi into Devanagari script. It is based on the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) and includes integration with [Weights & Biases (wandb)](https://wandb.ai/) for tracking experiments.

---

##  Project Structure

```
.
├── ma23m026_a3_da6401.py           # Contains model, training utilities, and helper functions
├── train_vanilla.py                  # Script to train vanilla Seq2Seq using argparse
├── predictions_without_attention.csv # Output predictions from the test set (generated)
├── README.md                         # This file
```

---

##  Requirements

Install the required Python packages:

```bash
pip install torch pandas matplotlib wandb
```

---

##  Dataset Requirements

Make sure you have the following dataset available:

- **Dakshina Dataset**:  
  `/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/`

The script reads training, validation, and test files from this location.

---

##  Running the Model

### Run with Default Settings

```bash
python train_vanilla.py
```

###  Custom Training Configuration

Override any training parameter via the command line:

```bash
python train_vanilla.py \
    --embed_dim 128 \
    --hidden_dim 256 \
    --enc_layers 2 \
    --dec_layers 2 \
    --cell_type GRU \
    --dropout 0.3 \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --teacher_forcing_ratio 0.6 \
    --optimizer nadam \
    --bidirectional \
    --pred_csv_path outputs/vanilla_preds.csv \
    --num_test_examples 100
```

---

##  Evaluation and Output

- Word-level accuracy is calculated on both validation and test sets.
- Final test predictions are saved to the path specified via `--pred_csv_path`.
- Training and validation metrics are logged to [wandb.ai](https://wandb.ai/).
-  By default, all 4502 test examples are evaluated.

---

##  Model Features

- Vanilla encoder-decoder architecture using configurable RNNs: `RNN`, `GRU`, or `LSTM`
- Support for bidirectional encoders via `--bidirectional` flag
- Command-line configurability with `argparse`
- Clean, modular codebase using reusable utilities
- Integrated logging and visualization with `wandb`

---

##  WandB Setup

Make sure to log in to Weights & Biases before training:

```bash
wandb login
```

Or set your API key in the script using:

```python
wandb.login(key='YOUR_WANDB_API_KEY')
```

---

## Output Files

| File                         | Description                                      |
|-----------------------------|--------------------------------------------------|
| `predictions_without_attention.csv` | Predictions on test set (by default all 4502 samples) |
| W&B Dashboard                | Tracks loss, accuracy, and hyperparameters       |

---

## Citation

If you use the Dakshina dataset, please cite the original creators. Refer to their [GitHub repo](https://github.com/google-research-datasets/dakshina) for citation details.

---

##  License

This project is provided for educational and research purposes only.
