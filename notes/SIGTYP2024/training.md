## Original Training
### Coptic `cop`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/cop_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/cop_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/cop/mlm" \
    --adapter_config "seq_bn" \
    --tokenizer_name "./custom_tokenizers/cop_tokenizer" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/cop_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/cop_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/cop/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --tokenizer_name "./custom_tokenizers/cop_tokenizer" \
      --mlm_adapter_path "./saved_models/cop/mlm" \
      --per_device_train_batch_size 36
```

#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/cop_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/cop_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/cop/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --tokenizer_name "./custom_tokenizers/cop_tokenizer" \
      --mlm_adapter_path "./saved_models/cop/mlm" \
      --per_device_train_batch_size 36
```

### Medieval Latin `latm`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/latm_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/latm_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/latm/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/latm_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/latm_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/latm/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/latm/mlm" \
      --per_device_train_batch_size 36
```

#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/latm_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/latm_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/latm/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/latm/mlm" \
      --per_device_train_batch_size 36
```

### Old French `fro`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/fro_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/fro_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/fro/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/fro_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/fro_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/fro/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/fro/mlm" \
      --per_device_train_batch_size 60
```

#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/fro_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/fro_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/fro/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/fro/mlm" \
      --per_device_train_batch_size 60
```

### Gothic `got`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/got_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/got_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/got/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/got_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/got_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/got/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/got/mlm" \
      --per_device_train_batch_size 60
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/got_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/got_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/got/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/got/mlm" \
      --per_device_train_batch_size 60
```

### Early Modern Irish `ghc`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/ghc_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/ghc_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/ghc/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```

### Ancient Greek `grc`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/grc_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/grc_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/grc/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/grc_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/grc_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/grc/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/grc/mlm" \
      --per_device_train_batch_size 36
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/grc_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/grc_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/grc/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/grc/mlm" \
      --per_device_train_batch_size 36
```

### Ancient Hebrew `hbo`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/hbo_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/hbo_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/hbo/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/hbo_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/hbo_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/hbo/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/hbo/mlm" \
      --per_device_train_batch_size 36
      
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/hbo_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/hbo_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/hbo/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/hbo/mlm" \
      --per_device_train_batch_size 36
```

### Medieval Icelandic `isl`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/isl_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/isl_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/isl/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/isl_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/isl_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/isl/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/isl/mlm" \
      --per_device_train_batch_size 36

```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/isl_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/isl_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/isl/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/isl/mlm" \
      --per_device_train_batch_size 36
```


### Middle Irish `mga`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/mga_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/mga_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/mga/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```

### Vedic Sanskrit `san`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/san_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/san_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/san/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/san_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/san_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/san/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/san/mlm" \
      --per_device_train_batch_size 36
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/san_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/san_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/san/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/san/mlm" \
      --per_device_train_batch_size 36
```

### Old Irish `sga`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/sga_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/sga_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/sga/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```


### Old East Slavic `orv`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/orv_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/orv_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/orv/mlm" \
    --adapter_config "seq_bn" \
    --tokenizer_name "./custom_tokenizers/orv_tokenizer" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/orv_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/orv_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/orv/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/orv/mlm" \
      --tokenizer_name "./custom_tokenizers/orv_tokenizer" \
      --per_device_train_batch_size 36
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/orv_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/orv_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/orv/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --tokenizer_name "./custom_tokenizers/orv_tokenizer" \
      --mlm_adapter_path "./saved_models/orv/mlm" \
      --per_device_train_batch_size 36
```

### Old Church Slavonic `chu`
#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/chu_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/chu_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/chu/mlm" \
    --adapter_config "seq_bn" \
    --tokenizer_name "./custom_tokenizers/chu_tokenizer" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/chu_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/chu_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/chu/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/chu/mlm" \
      --tokenizer_name "./custom_tokenizers/chu_tokenizer" \
      --per_device_train_batch_size 36
```
#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/chu_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/chu_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/chu/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --tokenizer_name "./custom_tokenizers/chu_tokenizer" \
      --mlm_adapter_path "./saved_models/chu/mlm" \
      --per_device_train_batch_size 36
```


### Classical Chinese `lzh`

```shell
export TRAIN_FILE=./converted_data/mlm/train/lzh_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/lzh_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/lzh/mlm" \
    --adapter_config "seq_bn" \
    --tokenizer_name "./custom_tokenizers/lzh_tokenizer" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```

#### Lemmatization
No model is trained for lemmatization

#### POS-tagging and Morphological Annotation
```shell
export TRAIN_FILE=./converted_data/tagging/train/lzh_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/lzh_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/lzh/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --tokenizer_name "./custom_tokenizers/lzh_tokenizer" \
      --mlm_adapter_path "./saved_models/lzh/mlm" \
      --per_device_train_batch_size 36
```
### Classical Chinese in decomposed form `lzh_decomposed`
#### Masked Language Modeling
Trained for character-level gap-filling
```shell
export TRAIN_FILE=./converted_data/mlm/train/lzh_decomposed_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/lzh_decomposed_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/lzh_decomposed/mlm" \
    --adapter_config "seq_bn" \
    --tokenizer_name "./custom_tokenizers/lzh_decomposed_tokenizer" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4
```

### Classical and Late Latin `lat`

#### Masked Language Modeling
```shell
export TRAIN_FILE=./converted_data/mlm/train/lat_train.txt
export VALIDATION_FILE=./converted_data/mlm/valid/lat_valid.txt

python train_mlm_adapter.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir "./saved_models/lat/mlm" \
    --adapter_config "seq_bn_inv" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4

```
#### Lemmatization
```shell
export TRAIN_FILE=./converted_data/lemmatisation/train/lat_train.json
export VALIDATION_FILE=./converted_data/lemmatisation/valid/lat_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "lemma_rules" \
      --output_dir "./saved_models/lat/lemmatisation" \
      --task_name "lemmatisation" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/lat/mlm" \
      --per_device_train_batch_size 36

```

#### POS-tagging and Morphological Annotation

```shell
export TRAIN_FILE=./converted_data/tagging/train/lat_train.json
export VALIDATION_FILE=./converted_data/tagging/valid/lat_valid.json

python train_token_classification_adapter.py \
      --model_name_or_path xlm-roberta-base \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --text_column_name "tokens" \
      --label_column_name "tags" \
      --output_dir "./saved_models/lat/tagging" \
      --task_name "tagging" \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --learning_rate 1e-3 \
      --num_train_epochs 10.0 \
      --save_total_limit 2 \
      --mlm_adapter_path "./saved_models/lat/mlm" \
      --per_device_train_batch_size 36
```