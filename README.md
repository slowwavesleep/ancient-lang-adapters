Steps to reproduce:

1. Clone the repository with shared task data
```
https://github.com/sigtyp/ST2024.git
```

2. Install requirements (in a virtual environment)
```
pip install -r requirements.txt
```

3. Convert data for training
```
python convert_lemmatisation.py
python convert_mlm.py
python convert_mlm_decomposed.py
python convert_tagging.py
```

4. Train the models

