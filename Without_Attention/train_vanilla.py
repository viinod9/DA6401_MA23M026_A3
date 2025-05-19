import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import wandb
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from ma23m026_a3_da6401 import (
    read_dataset, build_vocab, prepare_batch, calculate_word_accuracy,
    evaluate, Seq2Seq, predict_and_log_test_examples_with_csv
)

def main(args):
    wandb.init(config=args.__dict__, project="vanilla_attention_train")
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_data = read_dataset(args.train_path)
    dev_data = read_dataset(args.dev_path)
    test_path = args.test_path

    src_vocab, tgt_vocab = build_vocab([src for src, _ in train_data]), build_vocab([tgt for _, tgt in train_data])

    model = Seq2Seq(len(src_vocab[0]), len(tgt_vocab[0]), config.embed_dim, config.hidden_dim,
                    config.enc_layers, config.dec_layers, config.cell_type, config.dropout, config.bidirectional).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        random.shuffle(train_data)

        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i:i + config.batch_size]
            src, trg = prepare_batch(batch, src_vocab[0], tgt_vocab[0], device)

            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=config.teacher_forcing_ratio)

            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            acc = calculate_word_accuracy(output[:, 1:], trg[:, 1:])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / len(train_data)
        avg_train_acc = total_acc / (len(train_data) // config.batch_size)

        val_loss, val_acc = evaluate(model, dev_data, src_vocab[0], tgt_vocab[0], device, criterion, config.batch_size)

        wandb.log({
            "Train Loss": avg_train_loss,
            "Train Accuracy": avg_train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Run on test set
    predict_and_log_test_examples_with_csv(
        model,
        test_path,
        src_vocab,
        tgt_vocab,
        device,
        num_examples=args.num_test_examples,
        csv_save_path=args.pred_csv_path
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vanilla Seq2Seq model on Dakshina dataset")

    parser.add_argument('--train_path', type=str, default="/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    parser.add_argument('--dev_path', type=str, default="/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")
    parser.add_argument('--test_path', type=str, default="/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")
    parser.add_argument('--pred_csv_path', type=str, default="/kaggle/working/predictions_without_attention.csv")
    parser.add_argument('--num_test_examples', type=int, default=4502)

    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--cell_type', type=str, default='LSTM', choices=['RNN', 'GRU', 'LSTM'])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'nadam'])
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7)

    args = parser.parse_args()
    main(args)
