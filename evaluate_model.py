import torch
import sentencepiece as spm
import numpy as np
from torch.utils.data import DataLoader
from GRU import GRULanguageModel
from TextDataset import TextDataset
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from GRU.train_GRU import collate_fn
import argparse
import os

def compute_perplexity(model, data_loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(data_loader, desc="Computing Perplexity"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def compute_bleu_score(model, data_loader, tokenizer, device):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(data_loader, desc="Computing BLEU Score"):
            input_ids = input_ids.to(device)
            logits, _ = model(input_ids)
            predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            target_ids = target_ids.cpu().tolist()
            
            for pred, target in zip(predicted_ids, target_ids):
                pred_text = tokenizer.decode(pred)
                target_text = tokenizer.decode(target)
                
                references.append([target_text.split()])  # BLEU expects list of lists
                hypotheses.append(pred_text.split())
    
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the Perplexity and BLEU Score for a model")
    parser.add_argument("--model_path", type=str, default="GRU", help="Path to trained model file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(args.model_path, 'bpe_tokenizer.model'))
    vocab_size = tokenizer.get_piece_size()
    
    test_dataset = TextDataset("data/test.jsonl", tokenizer, 128)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    model = GRULanguageModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "best_model.pth"), map_location=device))
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=3)  # Ignore padding token
    
    ppl = compute_perplexity(model, test_loader, criterion, vocab_size, device)
    bleu = compute_bleu_score(model, test_loader, tokenizer, device)
    
    print(f"Perplexity: {ppl:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
