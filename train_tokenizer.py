import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=3_000)
    parser.add_argument("--tokenizers_dir", type=str, default="./custom_tokenizers")

    return parser.parse_args()


args = parse_arguments()
lang = args.lang
vocab_size = args.vocab_size
tokenizers_dir = Path(args.tokenizers_dir).resolve()
tokenizers_dir.mkdir(exist_ok=True, parents=True)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

with open(f"./converted_data/mlm/train/{lang}_train.txt") as file:
    train_sentences = file.readlines()

with open(f"./converted_data/mlm/valid/{lang}_valid.txt") as file:
    valid_sentences = file.readlines()

sentences = train_sentences + valid_sentences


def data_iterator():
    for sentence in sentences:
        yield sentence


custom_tokenizer = tokenizer.train_new_from_iterator(
    data_iterator(), vocab_size=vocab_size
)

custom_tokenizer.save_pretrained(tokenizers_dir / f"{lang}_tokenizer")
