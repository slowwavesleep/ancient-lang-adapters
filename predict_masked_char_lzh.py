import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import tqdm
from hanzipy.decomposer import HanziDecomposer


decomposer = HanziDecomposer()


def get_dict():
    data_dir = Path("./ST2024").resolve()
    data_path = data_dir / "fill_mask_char" / "train" / "lzh_train.tsv"
    data = pd.read_csv(data_path, sep="\t", quotechar="^")
    decomposer = HanziDecomposer()

    decomposed_list = []
    replacement_dict = {}
    for row in data["src"].tolist():
        cur_decomposed = ""
        for ch in row:
            components = decomposer.decompose(ch, 3)["components"]
            decomposed = "".join(components)
            cur_decomposed += decomposed
            replacement_dict[decomposed] = ch
        decomposed_list.append(cur_decomposed)
    return replacement_dict


def replace_substrings(input_string, replacement_dict):
    sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)

    index = 0

    result = ""

    replacements = dict()

    # Iterate through the string
    while index < len(input_string):
        replaced = False

        for key in sorted_keys:
            if input_string[index:].startswith(key):
                result += replacement_dict[key]
                index += len(key)
                replaced = True
                replacements[key] = replacement_dict[key]
                break

        # If no replacement was made, just add the current character to the result
        if not replaced:
            result += input_string[index]
            replacements[input_string[index]] = input_string[index]
            index += 1

    return result, replacements


repl_dict = get_dict()

submission_path = Path("./submission_3/fill_mask_char").resolve()
submission_path.mkdir(exist_ok=True, parents=True)

data_path = Path("./ST2024/fill_mask_char/test/lzh_test.tsv")
with open(data_path) as file:
    test_data = file.read().split("\n")
test_data = [el for el in test_data if el]


adapter_path = Path("./saved_models/lzh_decomposed/mlm")

root_adapter_path = Path(adapter_path).resolve()
mlm_adapter_path = root_adapter_path / "mlm"
mlm_adapter_config_path = mlm_adapter_path / "adapter_config.json"
mlm_adapter_config = AdapterConfig.load(mlm_adapter_config_path.as_posix())
model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    config=mlm_adapter_config,
)
model.load_adapter(mlm_adapter_path.as_posix(), config=mlm_adapter_config)
emb_path = root_adapter_path / "embeddings"
emb_pt_path = emb_path / "embedding.pt"
emb = torch.load(emb_pt_path.as_posix(), map_location=torch.device("cpu"))
torch.save(emb, emb_pt_path.as_posix())

model.load_embeddings(
    emb_path.as_posix(),
    "custom_embeddings",
)
model.set_active_adapters("mlm")
model.to("mps")

tokenizer = AutoTokenizer.from_pretrained(
    Path("./custom_tokenizers/lzh_decomposed_tokenizer").resolve().as_posix(),
    local_files_only=True,
)


def find_all_indices(input_string, ch):
    return [i for i, letter in enumerate(input_string) if letter == ch]


mask_token = tokenizer.mask_token
output = []
for index, row in tqdm(enumerate(test_data), total=len(test_data)):
    cur_candidates = []
    cur_possible_characters = []
    masked_sentence = row
    masked_sentence_copy = masked_sentence
    masked_sentence_copy = masked_sentence_copy.replace("[_]", ".")
    n_masked = masked_sentence.count("[_]")
    cur_result = {"masked": masked_sentence}
    while "[_]" in masked_sentence:
        cur_masked_sentence = masked_sentence.replace("[_]", mask_token, 1)
        tokenized_input = tokenizer(
            cur_masked_sentence, return_tensors="pt", truncation=True
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
            .argsort(descending=True, dim=-1)[:2000]
            .detach()
            .tolist()
        )
        possible_tokens = [
            token
            for token in tokenizer.convert_ids_to_tokens(possible_token_ids)
            if token != "▁"
            and token not in tokenizer.special_tokens_map.values()
            and token
        ]
        token_candidates = possible_tokens[:3]
        cur_candidates.append(token_candidates)
        masked_sentence = masked_sentence.replace("[_]", token_candidates[0], 1)
    composed, replacements = replace_substrings(masked_sentence, repl_dict)
    cur_result |= {"text": composed}
    masked_tokens = []
    if composed == "".join(replacements.values()):
        if len("".join(replacements.keys())) == len(masked_sentence_copy):
            cur_index = 0
            for key, value in replacements.items():
                if "." in masked_sentence_copy[cur_index : cur_index + len(key)]:
                    masked_tokens.append(value)
                cur_index += len(key)
    if len(masked_tokens) > n_masked:
        masked_tokens = masked_tokens[:n_masked]
    elif len(masked_tokens) < n_masked:
        masked_tokens = masked_tokens + [""] * (n_masked - len(masked_tokens))
    masked_tokens = [[el, "", ""] for el in masked_tokens]
    cur_result |= {"text": composed, "masked_tokens": masked_tokens}  # rerun
    output.append(cur_result)


with open(submission_path / "lzh.json", "w") as file:
    file.write(json.dumps(output, ensure_ascii=False))
