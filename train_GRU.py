from typing import Tuple
import os
import torch
import sentencepiece as spm
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from GRULanguageModel import GRULanguageModel
from TextDataset import TextDataset

BATCH_SIZE = 32
EPOCHS = 10

def add_special_tokens(pairs: Tuple[list]):
    """
    Insert <bos> and <eos> special tokens into a dataset
    :param pairs: original prompts and completions
    :return: prompts/completion pairs with special tokens inserted
    """
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(pairs):
        # If the beginning of the promot is upper case, then we assume it is the start of a sequence
        if prompt[0].isupper():
            prompt = '<bos>' + prompt

        # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
        if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
            completion += '<eos>'

        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions

# Merge all text files into a single corpus
def merge_text_files(data_dir, output_file):
    """
    This will merge all textual data in a directory into a single corpus
    :param data_dir: path to the directory containing the raw text files
    :param output_file: path to file where corpus will be saved
    """
    # open new file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in sorted(os.listdir(data_dir)):  # Sort for consistency
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")  # Ensure separation

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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else ("npm" if torch.npm.is_available() else "cpu"))
    # load in our tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.get_piece_size() # this gets the vocabulary size

    # load in the training and validation datasets
    train_dataset = TextDataset("data/train.jsonl", tokenizer, 128)
    val_dataset = TextDataset("data/test.jsonl", tokenizer, 128)

    # This will handle batching and shuffling during training. collate_fn handles padding of uneven sequences
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Instatiate our model and move it to the correct device memory (correcty device memory!!)
    model = GRULanguageModel(vocab_size=vocab_size).to(device)

    # Using AdamW optimizer on the trainable params. This is a stardard for LMs
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # going to use a learning rate scheduler that reduces LR by half after stagnation for 1 epch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    best_val_loss = float('inf') # keep track of best validation loss
    no_imporve_epochs = 0 # keep track of the number of epochs without any improvement
    # this will store the train and validation loss curves
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS): # loop over epochs
        model.train() # set model to training mode
        total_train_loss = 0 # keep track of training loss total
        # loop through each sample batch in training
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # move input and target tensors to device memory
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            # reset gradients between batches
            optimizer.zero_grad()
            # compute output logits
            logits, _ = model(input_ids)
            # apply cross entropy
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward() # backpropagate loss
            optimizer.step() # adjust trainable params

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")


if __name__ == "__main__":
    DATA_DIR = "data/raw" # path to raw data directory
    TOKENIZER_PREFIX = "bpe_tokenizer" # this will be used for naming the tokenizer
    VOCAB_SIZE = 10000 # stopping condition for tokenizing
    CORPUS_FILE = "corpus.txt" # path to the new combined corpus file
    merge_text_files(DATA_DIR, CORPUS_FILE)

    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        bos_id=1, # this is set to 1 because 0 is <unk>
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )

    print("Tokenizer training complete! Files generated:")
    print(f"- {TOKENIZER_PREFIX}.model")
    print(f"- {TOKENIZER_PREFIX}.vocab")

    train_model()

    