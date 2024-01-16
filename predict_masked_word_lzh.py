import json
from pathlib import Path

from transformers import AutoTokenizer
from adapters import AdapterConfig, AutoAdapterModel
import torch
from tqdm import tqdm


submission_path = Path("./submission_5/fill_mask_word").resolve()
submission_path.mkdir(exist_ok=True, parents=True)

data_path = Path("./ST2024/fill_mask_word/test/lzh_test.tsv")
with open(data_path) as file:
    test_data = file.read().split("\n")
test_data = [el for el in test_data if el]


adapter_path = Path("./saved_models_2/lzh/mlm")

root_adapter_path = Path(adapter_path).resolve()
mlm_adapter_path = root_adapter_path / "mlm"
mlm_adapter_config_path = mlm_adapter_path / "adapter_config.json"
mlm_adapter_config = AdapterConfig.load(mlm_adapter_config_path.as_posix())

model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    # config=mlm_adapter_config,
)

emb_path = root_adapter_path / "embeddings"
emb_pt_path = emb_path / "embedding.pt"
emb = torch.load(emb_pt_path.as_posix(), map_location=torch.device("cpu"))
torch.save(emb, emb_pt_path.as_posix())
model.load_embeddings(
    emb_path.as_posix(),
    "custom_embeddings",
)

model.load_adapter(mlm_adapter_path.as_posix(), config=mlm_adapter_config)
model.set_active_adapters("mlm")
model.to("mps")

tokenizer = AutoTokenizer.from_pretrained(
    Path("./custom_tokenizers/lzh_tokenizer").resolve().as_posix(),
    local_files_only=True,
)

mask_token = tokenizer.mask_token
output = []
for index, row in tqdm(enumerate(test_data), total=len(test_data)):
    cur_candidates = []
    masked_sentence = row
    masked_sentence_copy = masked_sentence
    n_masked = masked_sentence.count("[MASK]")
    cur_result = {"masked": masked_sentence_copy}
    while "[MASK]" in masked_sentence:
        cur_masked_sentence = masked_sentence.replace("[MASK]", mask_token, 1)
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
            if token != "▁" and token not in tokenizer.special_tokens_map.values()
        ][:3]
        cur_candidates.append(possible_tokens)
        masked_sentence = masked_sentence.replace("[MASK]", possible_tokens[0], 1)
    cur_result |= {"text": masked_sentence, "masked_tokens": cur_candidates}
    output.append(cur_result)


with open(submission_path / "lzh.json", "w") as file:
    file.write(json.dumps(output, ensure_ascii=False))
