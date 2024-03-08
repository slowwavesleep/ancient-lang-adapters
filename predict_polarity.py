
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
import adapters.composition as ac
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import trange

# model_name = "best-heur"
# run_num = 1
model_name = "best-synth"
run_num = 2


polarity2id = {"negative": 0, "positive": 1, "neutral": 2, "mixed": 3}
id2polarity = {value: key for key, value in polarity2id.items()}


def predict(input_data, batch_size):
    output = []
    for i in trange(0, len(input_data), batch_size):
        cur_batch = input_data[i: i + batch_size]
        tokenized = tokenizer(
            cur_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            preds = model(**tokenized).logits
        output.extend(preds.argmax(dim=-1).tolist())
    return output


model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

root_adapter_path = Path(f"./hala_models/{model_name}").resolve()
mlm_adapter_path = root_adapter_path / "mlm"
mlm_adapter_config_path = mlm_adapter_path / "adapter_config.json"
mlm_adapter_config = AdapterConfig.load(mlm_adapter_config_path.as_posix())

model.load_adapter(mlm_adapter_path.as_posix(), config=mlm_adapter_config)

polarity_adapter_path = root_adapter_path / "polarity"
polarity_adapter_config_path = polarity_adapter_path / "adapter_config.json"
polarity_adapter_config = AdapterConfig.load(
    polarity_adapter_config_path.as_posix()
)
model.load_adapter(
    polarity_adapter_path.as_posix(),
    config=polarity_adapter_config,
)


model.active_adapters = ac.Stack("mlm", "polarity")

model.to("mps")

submission_dir = Path(f"./{model_name}-submission").resolve()
submission_dir.mkdir(exist_ok=True)

for path in Path("emotion-test-data").resolve().iterdir():
    cur_write_path = submission_dir / f"emotion_{path.stem}_TartuNLP_{run_num}.tsv"
    cur_data = pd.read_csv(path, sep="\t", header=None, names=["sent_id", "text"])
    cur_texts = cur_data["text"].tolist()
    predicted_labels = [id2polarity[el] for el in predict(cur_texts, 8)]
    cur_data["labels"] = predicted_labels
    cur_data.to_csv(cur_write_path, sep="\t", index=False, header=False)
