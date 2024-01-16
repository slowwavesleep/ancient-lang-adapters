from pathlib import Path

import pandas as pd
from tqdm import tqdm


data_dir = Path("./ST2024").resolve()
output_dir = Path("./converted_data").resolve()

langs = [
    path.stem.split("_")[0]
    for path in (data_dir / "fill_mask_word" / "train").iterdir()
    if path.is_file()
]

for lang in tqdm(langs):
    for split in ("train", "valid"):
        data_path = data_dir / "fill_mask_word" / split / f"{lang}_{split}.tsv"
        file_path = output_dir / "mlm" / split / f"{lang}_{split}.txt"

        data = pd.read_csv(data_path, sep="\t", quotechar="^")

        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as file:
            file.write("\n".join(data["src"].tolist()))
