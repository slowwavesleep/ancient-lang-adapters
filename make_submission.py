from pathlib import Path

data_path = Path("./ST2024").resolve()
morphology_path = data_path / "morphology" / "test"
masked_langs = data_path / "fill_mask_word" / "test"

submission_path = Path("./submission_1")

morphology_langs = [
    p.stem.split("_")[0] for p in morphology_path.iterdir() if p.is_file()
]
masked_langs = [p.stem.split("_")[0] for p in masked_langs.iterdir() if p.is_file()]

for path in submission_path.iterdir():
    if not path.stem.startswith(".") and path.is_dir():
        submitted_langs = [p.stem.split("_")[0] for p in path.iterdir() if p.is_file()]
        if "mask" in path.stem:
            expected_langs = masked_langs
        else:
            expected_langs = morphology_langs
        if not set(submitted_langs) == set(expected_langs):
            missing = set(expected_langs) - set(submitted_langs)
            print(path)
            raise RuntimeError(f"Missing {missing} in {path.as_posix()}")
