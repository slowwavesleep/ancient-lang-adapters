from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def analyze_tokens(texts, tokenizer):
    total_tokens = 0
    unk_tokens = 0
    token_freq = Counter()

    for text in texts:
        encoded_text = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(encoded_text)
        unk_tokens += sum(
            [1 for token_id in encoded_text if token_id == tokenizer.unk_token_id]
        )
        token_freq.update(encoded_text)

    return total_tokens, unk_tokens, token_freq


MODEL = "xlm-roberta-base"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained("custom_tokenizer", local_files_only=True)
data_dir = Path("./ST2024").resolve()

langs = [
    path.stem.split("_")[0]
    for path in (data_dir / "fill_mask_word" / "train").iterdir()
    if path.is_file()
]

langs = ["cop"]

for lang in tqdm(langs):
    data_path = data_dir / "fill_mask_word" / "train" / f"{lang}_train.tsv"
    data = pd.read_csv(data_path, sep="\t", quotechar="^")

    texts = data["src"].tolist()

    total_tokens, unk_tokens, token_freq = analyze_tokens(texts, tokenizer)

    print(lang)
    print(f"Total tokens: {total_tokens}")
    print(f"Unknown tokens: {unk_tokens}")
    print(
        f"Frequency of some common tokens: {[(tokenizer.convert_ids_to_tokens(tok), num) for tok, num in token_freq.most_common(10)]}"
    )

    if total_tokens > 0:
        print(f"Percentage of unknown tokens: {100 * unk_tokens / total_tokens:.2f}%")
    else:
        print("No tokens found.")
