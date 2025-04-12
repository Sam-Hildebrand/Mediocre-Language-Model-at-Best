# Setup
Please place the data directory in the root of this project. e.g. `Project2/data`.

The Tokenizer is **already trained**, but, should you wish to retrain it, first `cd Tokenizer`, then run `python3 train_tokenizer.py`
This also formats the dataset, adding `<bos>` and `<eos>` symbols to the training and testing dataset, and creating `corpus.txt`

# To generate text from a model
From the root project folder (e.g. `Project2`), run `python3 generate_text.py --model_path [model] --max_length [max length] --temperature [temperature]`

- `[model]` can be one of `GRU`, `RNN`, `LSTM`, or `Transformer`
- `[max length]` can be an int `0` and above.
- `[temerature]` can be a float `0.0` and above. Set to `0` for greedy decoding.

# To Compute Perplexity and BLEU Scores for a Model
From the root project folder, run `python3 evaluate_model.py --model_path [model]` Where `[model]` is one of `GRU`, `RNN`, `LSTM`, or `Transformer`.

# Training
Should you wish to retrain a model, follow the steps in **Setup** then from the root project folder, run `python3 train_model.py --model_path [model]` where `[model]` can be one of `GRU`, `RNN`, `LSTM`, or `Transformer`.     