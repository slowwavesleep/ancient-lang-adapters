from pathlib import Path
import json

import conllu
from tqdm import tqdm

from lemma_rules import gen_lemma_rule

data_dir = Path("./ST2024").resolve()
output_dir = Path("./converted_data").resolve()
output_dir.mkdir(exist_ok=True)

langs = [
    path.stem.split("_")[0]
    for path in (data_dir / "morphology" / "train").iterdir()
    if path.is_file()
]


for lang in tqdm(langs):
    for split in ("train", "valid", "test"):
        data_path = data_dir / "morphology" / split / f"{lang}_{split}.conllu"
        file_path = output_dir / "lemmatisation" / split / f"{lang}_{split}.json"

        with open(data_path) as file:
            data_file = file.read()

        token_list = list(conllu.parse(data_file))
        output = []
        for id_, sent in enumerate(token_list):
            if "sent_id" in sent.metadata:
                idx = sent.metadata["sent_id"]
            else:
                idx = id_

            tokens = [token["form"] for token in sent]
            lemmas = [token["lemma"] for token in sent]
            lemma_rules = []
            for token, lemma in zip(tokens, lemmas):
                lemma_rules.append(gen_lemma_rule(token, lemma, allow_copy=True))

            if "text" in sent.metadata:
                txt = sent.metadata["text"]
            else:
                txt = " ".join(tokens)

            cur_element = {
                "idx": str(idx),
                "text": txt,
                "tokens": tokens,
                "lemmas": lemmas,
                "lemma_rules": lemma_rules,
            }
            output.append(cur_element)

        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as file:
            file.write(json.dumps(output, ensure_ascii=False))
