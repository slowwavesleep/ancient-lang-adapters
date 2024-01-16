import argparse
from collections import defaultdict
import json
from pathlib import Path


from transformers import AutoTokenizer
import adapters
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import trange
import numpy as np


def predict(input_data, pred_batch_size):
    tags_output = []
    pos_output = []
    for i in trange(0, len(input_data), pred_batch_size):
        tokens_list = [el["tokens"] for el in input_data[i : i + pred_batch_size]]
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
            pos_list = []
            feats_list = []
            for word_index, (item, word_id) in enumerate(
                zip(pred, tokenized[index].word_ids)
            ):
                if word_id is not None and word_id != prev:
                    predicted_labels.append(
                        id_to_label[pred[word_index, :].argmax().item()]
                    )
                prev = word_id
            for label, token in zip(predicted_labels, tokens):
                cur_feats = [el.split("=") for el in label.split(";")]
                cur_feats = {key: value for key, value in cur_feats}
                feats_list.append(cur_feats | {"Token": token})
                cur_pos = cur_feats["UPOS"]
                pos_list.append(cur_pos)
            tags_output.append(feats_list)
            pos_output.append(
                [[token, cur_pos] for cur_pos, token in zip(pos_list, tokens)]
            )
    return tags_output, pos_output


def evaluate_pos(input_data, ground_truth):
    hits_at_1 = 0
    total = 0
    for i, sent in enumerate(input_data):
        for pred, truth in zip(sent, ground_truth[i]["tags"]):
            cur_feats = [el.split("=") for el in truth.split(";")]
            cur_feats = {key: value for key, value in cur_feats}
            if cur_feats["UPOS"] == pred[1]:
                hits_at_1 += 1
            total += 1
    print(f"Validation POS accuracy @ 1: {hits_at_1 / total}")


def evaluate_tags(input_data, ground_truth):
    hits_at_1 = defaultdict(lambda: 0)
    total = defaultdict(lambda: 0)
    for i, sent in enumerate(input_data):
        for pred, truth in zip(sent, ground_truth[i]["tags"]):
            cur_feats = [el.split("=") for el in truth.split(";")]
            cur_feats = {key: value for key, value in cur_feats}
            missing = set(cur_feats) - set(pred)
            for key in pred.keys():
                if pred[key] == cur_feats.get(key, None):
                    hits_at_1[key] += 1
                total[key] += 1
            for key in missing:
                total[key] += 1

    accuracies = []
    for key in total:
        accuracies.append(hits_at_1[key] / total[key])
    print(f"Validation feats macro-average accuracy @ 1: {np.mean(accuracies)}")


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

pos_submission_dir = (Path(args.submission_dir) / "pos_tagging").resolve()
pos_submission_dir.mkdir(exist_ok=True, parents=True)
tagging_submission_dir = (Path(args.submission_dir) / "morph_features").resolve()
tagging_submission_dir.mkdir(exist_ok=True, parents=True)

data_path = Path("./converted_data/tagging").resolve()

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

tagging_adapter_path = root_adapter_path / "tagging"
tagging_adapter_config_path = tagging_adapter_path / "adapter_config.json"
tagging_adapter_config = AdapterConfig.load(tagging_adapter_config_path.as_posix())
model.load_adapter(
    tagging_adapter_path.as_posix(),
    config=tagging_adapter_config,
)


model.active_adapters = adapters.composition.Stack("mlm", "tagging")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model.to(device)

with open(Path(args.adapter_path) / "tagging_labels.json") as file:
    labels = json.loads(file.read())
id_to_label = {int(value): key for key, value in labels.items()}

with open(data_path / "valid" / f"{lang}_valid.json") as file:
    valid_data = json.loads(file.read())

with open(data_path / "test" / f"{lang}_test.json") as file:
    test_data = json.loads(file.read())

batch_size = args.batch_size


valid_tags, valid_pos = predict(valid_data, batch_size)


evaluate_pos(valid_pos, valid_data)
evaluate_tags(valid_tags, valid_data)

test_tags, test_pos = predict(test_data, batch_size)

with open(pos_submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(test_pos, ensure_ascii=False))

with open(tagging_submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(test_tags, ensure_ascii=False))
