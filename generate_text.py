import torch
import sentencepiece as spm
import argparse
from GRU import GRULanguageModel
import os

def generate_text(prompt, model, tokenizer, device, max_length=100):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    hidden = None
    
    generated_tokens = input_ids.clone().detach().tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(input_ids, hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            
            if next_token == tokenizer.eos_id():
                break
            
            generated_tokens.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(device)
    
    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained GRU language model.")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--model_path", type=str, default="GRU", help="Path to the model folder")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(args.model_path, "bpe_tokenizer.model"))
    vocab_size = tokenizer.get_piece_size()
    
    model = GRULanguageModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "best_model.pth"), map_location=device))
    
    generated_text = generate_text(args.prompt, model, tokenizer, device, args.max_length)
    print("Generated Text:")
    print(generated_text)
