import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import datetime
from pathlib import Path
from TextDataset import TextDataset, collate_fn

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 3  # Early stopping patience

def plot_loss_curve(train_losses, val_losses, model_name):
    os.makedirs("Graphs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"Graphs/{model_name}.png")
    plt.close()

def train_model(model, model_name):
    assert Path("data/train_processed.jsonl").exists(), "Training dataset not found!"
    assert Path("data/test_processed.jsonl").exists(), "Validation dataset not found!"
    
    train_dataset = TextDataset("data/train_processed.jsonl", tokenizer, 128)
    val_dataset = TextDataset("data/test_processed.jsonl", tokenizer, 128)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = f"{model_name}/best_{model_name}_model.pth"
            os.makedirs(model_name, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
            no_improve_epochs = 0
        else:
            print(f"No improvement. Remaining patience: {PATIENCE - no_improve_epochs} epochs")
            no_improve_epochs += 1
        
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    plot_loss_curve(train_losses, val_losses, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--model_path", type=str, choices=["GRU", "RNN", "LSTM", "Transformer"], required=True, help="Model architecture")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    
    tokenizer = spm.SentencePieceProcessor(model_file='Tokenizer/bpe_tokenizer.model')
    vocab_size = tokenizer.get_piece_size()
    
    model_classes = {
        "GRU": "GRULanguageModel",
        "RNN": "RNNLanguageModel",
        "LSTM": "LSTMLanguageModel",
        "Transformer": "TransformerLanguageModel"
    }
    
    module = __import__(args.model_path, fromlist=[model_classes[args.model_path]])
    ModelClass = getattr(module, model_classes[args.model_path])
    model = ModelClass(vocab_size=vocab_size).to(device)
    
    train_model(model, args.model_path)
