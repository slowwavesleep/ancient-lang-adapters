from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hanzipy.decomposer import HanziDecomposer


data_dir = Path("./ST2024").resolve()
output_dir = Path("./converted_data").resolve()

langs = ["lzh"]

decomposer = HanziDecomposer()

for lang in tqdm(langs):
    for split in ("train", "valid"):
        data_path = data_dir / "fill_mask_char" / split / f"{lang}_{split}.tsv"
        file_path = output_dir / "mlm" / split / f"{lang}_decomposed_{split}.txt"

        data = pd.read_csv(data_path, sep="\t", quotechar="^")
        lines = []
        for row in data["src"].tolist():
            cur_decomposed = []
            for ch in row:
                components = decomposer.decompose(ch, 3)["components"]
                cur_decomposed.extend(components)
            lines.append("".join(cur_decomposed))

        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as file:
            file.write("\n".join(lines))
