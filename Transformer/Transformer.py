import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from BaseLanguageModel import BaseLanguageModel

class TransformerLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embed_dim=256, num_layers=6, nhead=2, max_seq_length=512, dropout=0.2, pad_token_id=3):
        super().__init__(vocab_size, embed_dim, pad_token_id, model_name="Transformer")
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        Compute model output logits given a sequence
        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        token_embeddings = self.embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.pos_embedding(positions)
        embeds = self.dropout(token_embeddings + pos_embeddings)

        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input_ids.device)
        transformer_output = self.transformer_encoder(embeds, mask=mask)
        
        return self.fc(transformer_output), None
