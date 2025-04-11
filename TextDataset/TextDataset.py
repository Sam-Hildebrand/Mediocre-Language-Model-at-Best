import torch
from torch.utils.data import Dataset
import json
from torch import nn

def collate_fn(batch):
    """
    Ensure batch is appropriately sized and padded for efficeient training
    :param batch: batch from DataLoader, which will be a list of Tuples of token ID tensors
        (which could be different sizes)
    :return: collated inputs and target batch
    """
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        """
        Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions
        for Causal LM
        :param filepath: path to the jsonl file
        :param tokenizer: instance of trained SentencePiece tokenizer
        :param max_seq_len: maximum sequence length we want to allow
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # open the jsonl file and tokenize each samples
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # we are using Causal Language Modeling, prompts and completions treated the same way!
                #text = item["prompt"] + " " + item["completion"]
                text = item.get("prompt", "") + " " + item.get("completion", "")
                # tokenize the fill prompt + completion (truncate at max sequence length)
                token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]
                # make sure we don't have any overly short samples
                if len(token_ids) < 2:
                    continue
                # append tokenized sample to list
                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get and format samples at the given index
        For Causal Language Modeling, we will train the model to predict every next token in the sequence given
        the prior ones
        :param idx:
        :return:
        """
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
    