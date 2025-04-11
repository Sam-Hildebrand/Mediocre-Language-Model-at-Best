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

class RNNLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=3):
        """
        Recurrent Neural Network (RNN) Language Model
        :param vocab_size: Vocabulary size
        :param embed_dim: Size of token embedding vectors
        :param hidden_dim: Hidden size of the RNN
        :param num_layers: Number of RNN layers
        :param dropout: Dropout rate
        :param pad_token_id: Token ID of <pad> token
        """
        super().__init__(vocab_size, embed_dim, pad_token_id, model_name="RNN")
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Compute model output logits given a sequence
        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids)
        output, hidden = self.rnn(embeds, hidden)
        return self.fc(output), hidden