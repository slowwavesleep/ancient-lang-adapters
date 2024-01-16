import argparse
from pathlib import Path

from collections import defaultdict
from itertools import chain
import re
import json

from tqdm import tqdm
import pandas as pd

from utils import process_list


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--submission_dir", type=str, required=True)
    return parser.parse_args()


def match_word(pattern: str, length_map: dict[int, list[str]]) -> list[str]:
    if not pattern:
        return []
    pattern = pattern.replace("[_]", ".")
    pattern = ".".join([re.escape(el) for el in pattern.split(".")])
    regex_pattern = re.compile(pattern)
    candidates = length_map[len(pattern)]
    matched_words = [
        candidate for candidate in candidates if re.fullmatch(regex_pattern, candidate)
    ]
    return matched_words


def rank_candidates(candidates_list: list[str]) -> list[str]:
    if len(candidates_list) < 2:
        return candidates_list
    else:
        return candidates_list


def find_hidden_chars(original, masked):
    hidden_chars = []
    masked_index = 0

    for char in original:
        if masked_index < len(masked) and (
            char == masked[masked_index] or masked[masked_index] != "["
        ):
            masked_index += 1
        elif masked[masked_index] == "[":
            hidden_chars.append(char)
            # Skip the '[_]' sequence
            masked_index += 3

    return hidden_chars


def extract_matched_characters(word_with_gaps: str, candidate_words: list[str]):
    parts = word_with_gaps.split("[_]")
    num_gaps = len(parts) - 1

    # Initialize a list to store possible characters for each gap
    gap_characters = [[] for _ in range(num_gaps)]

    # Iterate over each candidate word
    for candidate in candidate_words:
        # Handle whitespace as a special case
        if candidate == " ":
            for gap_list in gap_characters:
                gap_list.append(" ")
            continue

        if len(candidate) != len(word_with_gaps) - num_gaps * len("[_]") + num_gaps:
            # Skip candidates that don't match the length of the word with gaps filled
            continue

        # Check if the candidate matches the pattern
        candidate_index = 0
        matches = True
        for i, part in enumerate(parts):
            index = candidate.find(part, candidate_index)

            if index != candidate_index:
                # Part doesn't match at the expected position
                matches = False
                break

            # Add the character filling the gap to the corresponding list
            if i < num_gaps:
                gap_characters[i].append(candidate[index + len(part)])

            # Update the index for the next iteration
            candidate_index = index + len(part) + 1

        if not matches:
            # Remove the characters added for this candidate, as it doesn't fit
            for gap_list in gap_characters:
                if gap_list and gap_list[-1] == candidate[candidate_index - 1]:
                    gap_list.pop()

    return gap_characters


def predict(data):
    output = []
    for vaL_el in tqdm(data):
        masked_sent = vaL_el["masked"]
        word_gaps = []
        cur_words = masked_sent.split()
        masked_words = []
        masked_word_ids = []
        word_char_gaps = []
        result = {"masked": masked_sent}
        for index, cur_word in enumerate(cur_words):
            if "[_]" in cur_word:
                masked_words.append(cur_word)
                masked_word_ids.append(index)
                matched = match_word(cur_word, words_map)
                cur_char_gap = cur_word.count("[_]")
                word_char_gaps.append(cur_char_gap)
                if not matched and cur_char_gap == 1:
                    left, right = cur_word.split("[_]")
                    left_match = left in words_map[len(left)]
                    right_match = right in words_map[len(right)]
                    if (
                        (left_match and not right)
                        or (right_match and not left)
                        or (left_match and right_match)
                    ):
                        word_gaps.append([" "])
                    else:
                        word_gaps.append(matched)
                else:
                    word_gaps.append(matched)
        predictions = None
        if word_gaps:
            char_gaps = []
            word_gaps = [rank_candidates(gap) for gap in word_gaps]
            for masked_word, candidate in zip(masked_words, word_gaps):
                cur_char_gaps = extract_matched_characters(masked_word, candidate)
                char_gaps.extend(cur_char_gaps)
            predictions = {
                "masked_tokens": [process_list(matches) for matches in char_gaps]
            }
            assert len(predictions["masked_tokens"]) == masked_sent.count("[_]")
            result |= predictions
        if predictions:
            restored_sent = masked_sent
            for ch in [pred[0] for pred in predictions["masked_tokens"]]:
                restored_sent = restored_sent.replace("[_]", ch, 1)
            result |= {"text": restored_sent}
        else:
            result |= {
                "masked_tokens": [[""] * 3 for _ in range(masked_sent.count("[_]"))]
            }
            result |= {"text": masked_sent}

        output.append(result)
    return output


def evaluate(validation, prediction):
    hits_at_1 = 0
    hits_at_3 = 0
    total = 0
    for val_el, pred_el in zip(validation, prediction):
        true_masked_tokens = [
            token["masked_token"] for token in val_el["masked_tokens"]
        ]
        predicted_masked_tokens = pred_el["masked_tokens"]

        for true_token, pred_tokens in zip(true_masked_tokens, predicted_masked_tokens):
            if true_token:
                if true_token in pred_tokens:
                    hits_at_3 += 1
                if true_token == pred_tokens[0]:
                    hits_at_1 += 1
                total += 1
    print(f"Accuracy @ 1: {hits_at_1 / total}")
    print(f"Accuracy @ 3: {hits_at_3 / total}")


args = parse_arguments()

lang = args.lang

submission_dir = (Path(args.submission_dir) / "fill_mask_char").resolve()
submission_dir.mkdir(exist_ok=True, parents=True)

data_path = Path("./ST2024/fill_mask_char").resolve()
train_data = pd.read_csv(
    data_path / "train" / f"{lang}_train.tsv", sep="\t", quotechar="^"
)
with open(data_path / "valid" / "json" / f"{lang}_valid.json") as file:
    valid_data = json.loads(file.read())

test_data = []
with open(data_path / "test" / f"{lang}_test.tsv") as file:
    for line in file:
        test_data.append({"masked": line.strip("\n")})

train_words = set(chain.from_iterable(train_data["src"].str.split()))
words_map = defaultdict(lambda: [])
for word in train_words:
    words_map[len(word)].append(word)


val_output = predict(valid_data)
evaluate(validation=valid_data, prediction=val_output)

test_output = predict(test_data)

with open(submission_dir / f"{lang}.json", "w") as file:
    file.write(json.dumps(test_output, ensure_ascii=False))
