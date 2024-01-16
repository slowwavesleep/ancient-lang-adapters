from pathlib import Path

from transformers import AutoTokenizer

lang = "lzh"

tokenizers_dir = Path("./custom_tokenizers").resolve()
tokenizers_dir.mkdir(exist_ok=True, parents=True)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

with open(f"./converted_data/mlm/train/{lang}_decomposed_train.txt") as file:
    train_sentences = file.readlines()

with open(f"./converted_data/mlm/valid/{lang}_decomposed_valid.txt") as file:
    valid_sentences = file.readlines()

sentences = train_sentences + valid_sentences

unique_chars = set()

for sentence in sentences:
    unique_chars.update(sentence)

vocab_size = len(unique_chars) + 20


def data_iterator():
    for sentence in sentences:
        yield sentence


custom_tokenizer = tokenizer.train_new_from_iterator(
    data_iterator(), vocab_size=vocab_size
)

custom_tokenizer.save_pretrained(tokenizers_dir / f"{lang}_decomposed_tokenizer")
