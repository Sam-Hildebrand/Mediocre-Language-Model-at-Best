from typing import Tuple
import os
import sentencepiece as spm
import json

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

def process_jsonl(file_path):
    """
    Process a JSONL file by applying special tokens.
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        pairs = [(json.loads(line)['prompt'], json.loads(line)['completion']) for line in infile]
    
    new_prompts, new_completions = add_special_tokens(pairs)
    
    for prompt, completion in zip(new_prompts, new_completions):
        processed_data.append({'prompt': prompt, 'completion': completion})
    
    output_file = file_path.replace('.jsonl', '_processed.jsonl')
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    print(f"Processed {file_path} and saved to {output_file}")

# Merge all text files into a single corpus
def merge_text_files(data_dir, output_file):
    """
    Merge all textual data in a directory into a single corpus with special tokens inserted.
    :param data_dir: path to the directory containing the raw text files
    :param output_file: path to file where corpus will be saved
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in sorted(os.listdir(data_dir)):  # Sort for consistency
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")  # Ensure separation


if __name__ == "__main__":
    DATA_DIR = "../data/raw" # path to raw data directory
    TOKENIZER_PREFIX = "bpe_tokenizer" # this will be used for naming the tokenizer
    VOCAB_SIZE = 10000 # stopping condition for tokenizing
    CORPUS_FILE = "../data/corpus.txt" # path to the new combined corpus file
    merge_text_files(DATA_DIR, CORPUS_FILE)

    print("Applying Special Tokens to Testing and Training Dataset")
    process_jsonl("../data/test.jsonl")
    process_jsonl("../data/train.jsonl")

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