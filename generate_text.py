import torch
import sentencepiece as spm
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained GRU language model.")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--model_path", type=str, default="GRU", help="Path to the model folder")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation Temperature")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    

    tokenizer = spm.SentencePieceProcessor("Tokenizer/bpe_tokenizer.model")
    vocab_size = tokenizer.get_piece_size()

    if args.model_path == "GRU":
        from GRU import GRULanguageModel
        model = GRULanguageModel(vocab_size=vocab_size).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "best_GRU_model.pth"), map_location=device))
    elif args.model_path == "RNN":
        from RNN import RNNLanguageModel
        model = RNNLanguageModel(vocab_size=vocab_size).to(device) 
        model.load_state_dict(torch.load(os.path.join(args.model_path, "best_RNN_model.pth"), map_location=device))
    elif args.model_path == "LSTM":
        from LSTM import LSTMLanguageModel
        model = LSTMLanguageModel(vocab_size=vocab_size).to(device) 
        model.load_state_dict(torch.load(os.path.join(args.model_path, "best_LSTM_model.pth"), map_location=device))
    elif args.model_path == "Transformer":
        from Transformer import TransformerLanguageModel
        model = TransformerLanguageModel(vocab_size=vocab_size).to(device) 
        model.load_state_dict(torch.load(os.path.join(args.model_path, "best_Transformer_model.pth"), map_location=device))


    generated_text = model.generate(tokenizer, args.prompt, temperature=args.temperature, device=device, max_length=args.max_length)

    print("Generated Text:")
    print(generated_text)
