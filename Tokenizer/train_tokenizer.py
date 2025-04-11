from typing import Tuple
import os
import sentencepiece as spm

def add_special_tokens(pairs: Tuple[list]):
    """
    Insert <bos> and <eos> special tokens into a dataset
    :param pairs: original prompts and completions
    :return: prompts/completion pairs with special tokens inserted
    """
    new_prompts = []
    new_completions = []

    for prompt, completion in pairs:
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
    Merge all textual data in a directory into a single corpus with special tokens inserted.
    :param data_dir: path to the directory containing the raw text files
    :param output_file: path to file where corpus will be saved
    """
    all_pairs = []
    
    for filename in sorted(os.listdir(data_dir)):  # Sort for consistency
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as infile:
                lines = [line.strip() for line in infile.readlines() if line.strip()]
                for i in range(0, len(lines) - 1, 2):  # Assuming even indices are prompts, odd are completions
                    prompt = lines[i]
                    completion = lines[i + 1] if i + 1 < len(lines) else ""
                    all_pairs.append((prompt, completion))
    
    # Apply special tokens
    new_prompts, new_completions = add_special_tokens(all_pairs)
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for prompt, completion in zip(new_prompts, new_completions):
            outfile.write(prompt + "\n" + completion + "\n")


if __name__ == "__main__":
    DATA_DIR = "../data/raw" # path to raw data directory
    TOKENIZER_PREFIX = "bpe_tokenizer" # this will be used for naming the tokenizer
    VOCAB_SIZE = 10000 # stopping condition for tokenizing
    CORPUS_FILE = "../data/corpus.txt" # path to the new combined corpus file
    merge_text_files(DATA_DIR, CORPUS_FILE)

    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        bos_id=1, # this is set to 1 because 0 is <unk>
        eos_id=2,
        pad_id=3,
        user_defined_symbols=["<pad>"]
    )

    print("Tokenizer training complete! Files generated:")
    print(f"- {TOKENIZER_PREFIX}.model")
    print(f"- {TOKENIZER_PREFIX}.vocab")