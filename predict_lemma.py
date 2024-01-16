import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
import adapters.composition as ac
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import trange

from lemma_rules import apply_lemma_rule


def predict(input_data, batch_size):
    output = []
    for i in trange(0, len(input_data), batch_size):
        tokens_list = [el["tokens"] for el in input_data[i : i + batch_size]]
        tokenized = tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            preds = model(**tokenized).logits
        for index, (pred, tokens) in enumerate(zip(preds, tokens_list)):
            prev = None
            predicted_labels = []
            for word_index, (item, word_id) in enumerate(
                zip(pred, tokenized[index].word_ids)
            ):
                if word_id is not None and word_id != prev:
                    predicted_labels.append(
                        [
                            apply_lemma_rule(tokens[word_id], id_to_label[id_])
                            for id_ in pred[word_index, :]
                            .argsort(descending=True)[:3]
                            .tolist()
                        ]
                    )
                prev = word_id
            output.append(
                [
                    [token, candidates]
                    for token, candidates in zip(tokens, predicted_labels)
                ]
            )
    return output


def evaluate(predicted_data, ground_truth, batch_size):
    hits_at_1 = 0
    hits_at_3 = 0
    total = 0
    for i, sent in enumerate(predicted_data):
        for pred, truth in zip(sent, ground_truth[i]["lemmas"]):
            if truth in pred[1]:
                hits_at_3 += 1
            if truth in pred[1][0]:
                hits_at_1 += 1
            total += 1

    print(f"Validation accuracy @ 1 for `{lang}` on lemmatisation: {hits_at_1 / total}")
    print(
        f"Validation accuracy @ 3: for `{lang}` on lemmatisation: {hits_at_3 / total}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--submission_dir", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


args = parse_arguments()

lang = args.lang

submission_dir = (Path(args.submission_dir) / "lemmatisation").resolve()
submission_dir.mkdir(exist_ok=True, parents=True)

if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.tokenizer_name).resolve().as_posix(),
        local_files_only=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

root_adapter_path = Path(args.adapter_path).resolve()
mlm_adapter_path = root_adapter_path / "mlm"
mlm_adapter_config_path = mlm_adapter_path / "adapter_config.json"
mlm_adapter_config = AdapterConfig.load(mlm_adapter_config_path.as_posix())
model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    # config=mlm_adapter_config,
)

if args.tokenizer_name:
    # stupid workaround to embeddings beings saved to cuda device
    emb_path = root_adapter_path / "embeddings"
    emb_pt_path = emb_path / "embedding.pt"
    emb = torch.load(emb_pt_path.as_posix(), map_location=torch.device("cpu"))
    torch.save(emb, emb_pt_path.as_posix())

    model.load_embeddings(
        emb_path.as_posix(),
        "custom_embeddings",
    )

model.load_adapter(mlm_adapter_path.as_posix(), config=mlm_adapter_config)


lemmatisation_adapter_path = root_adapter_path / "lemmatisation"
lemmatisation_adapter_config_path = lemmatisation_adapter_path / "adapter_config.json"
lemmatisation_adapter_config = AdapterConfig.load(
    lemmatisation_adapter_config_path.as_posix()
)
model.load_adapter(
    lemmatisation_adapter_path.as_posix(),
    config=lemmatisation_adapter_config,
)


model.active_adapters = ac.Stack("mlm", "lemmatisation")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.to(device)

with open(Path(args.adapter_path) / "lemmatisation_labels.json") as file:
    labels = json.loads(file.read())
id_to_label = {int(value): key for key, value in labels.items()}


data_dir = Path("./converted_data").resolve()
with open(data_dir / "lemmatisation" / "valid" / f"{lang}_valid.json") as file:
    valid_data = json.loads(file.read())
with open(data_dir / "lemmatisation" / "test" / f"{lang}_test.json") as file:
    test_data = json.loads(file.read())


valid_output = predict(valid_data, args.batch_size)
evaluate(valid_output, valid_data, args.batch_size)

test_output = predict(test_data, args.batch_size)
assert len(test_output) == len(test_data)

with open(submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(test_output, ensure_ascii=False))
