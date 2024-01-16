import argparse
from collections import defaultdict
import json
from pathlib import Path

import conllu

from utils import process_list


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, required=True)
    return parser.parse_args()


args = parse_arguments()
submission_dir = (Path(args.submission_dir) / "lemmatisation").resolve()
submission_dir.mkdir(exist_ok=True, parents=True)

data_dir = Path("./ST2024").resolve()

lang = "lzh"

train_path = data_dir / "morphology" / "train" / f"{lang}_train.conllu"
valid_path = data_dir / "morphology" / "valid" / f"{lang}_valid.conllu"
test_path = data_dir / "morphology" / "test" / f"{lang}_test.conllu"

with open(train_path) as file:
    train_file = file.read()


lemma_dict = defaultdict(lambda: [])
token_list = list(conllu.parse(train_file))
for sent in token_list:
    tokens = [token["form"] for token in sent]
    lemmas = [token["lemma"] for token in sent]
    for token, lemma in zip(tokens, lemmas):
        if lemma not in lemma_dict[token]:
            lemma_dict[token].append(lemma)


with open(valid_path) as file:
    valid_file = file.read()


token_list = list(conllu.parse(valid_file))
valid_lemmas = []
predicted_valid = []
for sent in token_list:
    tokens = [token["form"] for token in sent]
    lemmas = [token["lemma"] for token in sent]
    valid_lemmas.append(lemmas)
    cur_preds = []
    for token in tokens:
        cur_preds.append(lemma_dict.get(token, token))
    predicted_valid.append(cur_preds)

hits_at_1 = 0
hits_at_3 = 0
total = 0

for true_sent, pred_sent in zip(valid_lemmas, predicted_valid):
    for true_lemma, pred_lemma in zip(true_sent, pred_sent):
        if true_lemma == pred_lemma[0]:
            hits_at_1 += 1
        if true_lemma in pred_lemma[:3]:
            hits_at_3 += 1
        total += 1

print(hits_at_1 / total)
print(hits_at_3 / total)


with open(test_path) as file:
    test_file = file.read()

token_list = list(conllu.parse(test_file))

predicted_test = []
test_output = []
for sent in token_list:
    tokens = [token["form"] for token in sent]
    cur_preds = []
    for token in tokens:
        cur_preds.append([token, process_list(lemma_dict.get(token, [token]))])
    test_output.append(cur_preds)

assert len(token_list) == len(test_output)

with open(submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(test_output, ensure_ascii=False))
