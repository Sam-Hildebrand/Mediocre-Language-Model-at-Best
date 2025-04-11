import torch
from torch import nn
import torch.nn.functional as F

class BaseLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_token_id, model_name):
        """
        Base class for language models.
        :param vocab_size: Size of the vocabulary
        :param embed_dim: Embedding dimension for token representations
        :param pad_token_id: Token ID for the <pad> token
        :param model_name: Name of the model
        """
        super(BaseLanguageModel, self).__init__()
        self.name = model_name
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

    def predict_next_token(self, input_ids, temperature=1.0, bos_token_id=None):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input sequence token IDs
        :param temperature: setting for sampling
        :return: next token ID, hidden state
        """
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            
            if bos_token_id is not None:
                logits[:, bos_token_id] = -float('inf')  # Prevent <bos> from being sampled
            
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1) if temperature == 0 else torch.multinomial(probs, num_samples=1).squeeze(-1)

            return next_token_id.item(), hidden

    def generate(self, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
        """
        Generate text from a given prompt.
        """
        eos_token_id = tokenizer.eos_id()
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []
        hidden = None

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, temperature, bos_token_id=tokenizer.bos_id())
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        return tokenizer.decode(generated_ids, out_type=str)
