import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import tqdm


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

if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.tokenizer_name).resolve().as_posix(),
        local_files_only=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


model.set_active_adapters("mlm")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.to(device)


data_path = Path(f"ST2024/fill_mask_word/test/{lang}_test.tsv").resolve()
with open(data_path) as file:
    data = file.read().split("\n")
data = [el for el in data if el]


submission_dir = (Path(args.submission_dir) / "fill_mask_word").resolve()
submission_dir.mkdir(exist_ok=True, parents=True)


def predict_full_words(masked_sentence, tokenizer, model, max_subtokens_per_mask):
    mask_token = tokenizer.mask_token
    masked_sentence = masked_sentence.replace("[MASK]", mask_token, 1)
    tokenized_input = tokenizer(
        masked_sentence, return_tensors="pt", truncation=True
    ).to(model.device)
    input_ids = tokenized_input["input_ids"].tolist()[0]
    attention_mask = tokenized_input["attention_mask"].tolist()[0]
    mask_token_index = input_ids.index(tokenizer.mask_token_id)
    decoded_token_ids = []

    with torch.no_grad():
        outputs = model(**tokenized_input)
    predictions = outputs.logits

    possible_token_ids = (
        predictions[0, mask_token_index]
        .argsort(descending=True, dim=-1)[:3000]
        .detach()
        .tolist()
    )
    possible_tokens = tokenizer.convert_ids_to_tokens(possible_token_ids)
    best_token = None
    best_token_id = None
    for token, token_id in zip(possible_tokens, possible_token_ids):
        if (
            token.startswith("▁")
            and len(token) > 1
            and token not in tokenizer.special_tokens_map.values()
        ):
            best_token = token
            best_token_id = token_id
            break
    if not best_token or not best_token_id:
        raise ValueError("No suitable token", masked_sentence)
    input_ids[mask_token_index] = best_token_id
    decoded_token_ids.append(best_token_id)

    next_input_ids = list(input_ids)
    for i in range(1, max_subtokens_per_mask + 1):
        next_input_ids = list(input_ids)
        next_attention_mask = list(attention_mask)
        next_input_ids.insert(mask_token_index + i, tokenizer.mask_token_id)
        next_attention_mask.insert(mask_token_index + i, 1)

        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([next_input_ids]).to(model.device),
                attention_mask=torch.tensor([next_attention_mask]).to(model.device),
            )
        next_predictions = outputs.logits

        next_best_token_id = (
            next_predictions[0, mask_token_index + i].argmax(dim=-1).item()
        )
        next_best_token = tokenizer.convert_ids_to_tokens(next_best_token_id)
        if next_best_token.startswith("▁"):
            next_input_ids = list(input_ids)
            break
        else:
            next_input_ids[mask_token_index + i] = next_best_token_id
            decoded_token_ids.append(next_best_token_id)
    predicted_sentence = tokenizer.decode(next_input_ids, skip_special_tokens=True)
    decoded_word = tokenizer.decode(decoded_token_ids, skip_special_tokens=True)
    return predicted_sentence, decoded_word


output = []
for row in tqdm(data, total=len(data)):
    masked_sentence = row
    n_masked = masked_sentence.count("[MASK]")
    max_subtokens_per_mask = 1
    predicted_sentence = None
    predicted_words = []
    cur_output = {"masked": masked_sentence}
    while "[MASK]" in masked_sentence:
        masked_sentence, masked_word = predict_full_words(
            masked_sentence, tokenizer, model, max_subtokens_per_mask
        )
        predicted_words.append(masked_word)
    if not predicted_sentence:
        predicted_sentence = {"text": masked_sentence}
    else:
        predicted_sentence = {"text": predicted_sentence}
    cur_output = cur_output | predicted_sentence
    if predicted_words:
        if not n_masked == len(predicted_words):
            predicted_words = {"masked_tokens": [["", "", ""] for _ in range(n_masked)]}
        else:
            predicted_words = {
                "masked_tokens": [[word, "", ""] for word in predicted_words]
            }
    else:
        predicted_words = {"masked_tokens": [["", "", ""] for _ in range(n_masked)]}
    cur_output = cur_output | predicted_words
    output.append(cur_output)

with open(submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(output, ensure_ascii=False))
