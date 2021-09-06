
# How to use (mC4+wiki40b)

## Setup TPU

- Create an instance to work there
    - [Preventing accidental VM deletion](https://cloud.google.com/compute/docs/instances/preventing-accidental-vm-deletion) by typing ``ctpu delete``
- [Install ctpu](https://github.com/tensorflow/tpu/tree/master/tools/ctpu#local-machine)
    - Download, make runnable with ``chmod`` and place it under ``$PATH``
- You should add the permission of the cloud storage for the service account.

    ```bash
    gsutil iam ch serviceAccount:service-XXXXXXXXXX@cloud-tpu.iam.gserviceaccount.com:roles/storage.objectAdmin gs://YOUR_GSC_BUCKET
    ```

## Extract Japanese Wikipedia texts

```bash
mkdir -p ~/data/wiki-40b/plain
poetry run python3 ./scripts/setup_wikipedia.py --lang ja | gzip > ~/data/wiki-40b/plain/wiki-40b.ja.gz
```

I took about 4 minutes.
The resulting file contained 2,073,584 articles.

## Train SentencePiece with Japanese Wikipedia

Install [SentencePiece](https://github.com/google/sentencepiece).

```bash
mkdir -p ~/data/wiki-40b/spm/
zcat ~/data/wiki-40b/plain/wiki-40b.ja.gz > ~/data/wiki-40b/spm/sent.spm.txt
mkdir -p ~/data/wiki-40b/spm/spm
spm_train -vocab_size 32000 \
    --model_type unigram \
    --model_prefix ~/data/wiki-40b/spm/spm/spm4bert \
    --input ~/data/wiki-40b/spm/sent.spm.txt \
    --num_threads $(nproc) \
    --control_symbols='' \
    --input_sentence_size 3000000 \
    --shuffle_input_sentence=true \
    --add_dummy_prefix=false \
    --byte_fallback --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1
gsutil cp ~/data/wiki-40b/spm/spm/spm4bert.model gs://YOUR_GSC_BUCKET/spm/spm.wiki40b-ja.32k.byte_fallback.forT5.model
```

I took about 30 minutes with 16 CPU cores.

## TFDS preparation

### Japanese Wikipedia

```bash
poetry run python3 ./scripts/gen.py -s mywiki40b -i ~/data/wiki-40b/plain/ -o ~/data/wiki-40b/tfds/
gsutil -m cp -r ~/data/wiki-40b/tfds/mywiki40b gs://YOUR_GSC_BUCKET/tfds/mywiki40b
```

### Japanese part of mC4

- Be aware [this may cost much](https://github.com/allenai/allennlp/discussions/5056)
    - Copy to a bucket in ``us-central1`` cost about 900 yen (Inter-region GCP Storage egress within NA for 810GB)

```bash
gsutil -u YOUR_PROJECT_NAME -m cp \
    'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/c4-ja-validation.*' \
    'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/c4-ja.*' \
    'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/dataset_info.json' \
    'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/features.json'\
    gs://YOUR_GSC_BUCKET/tfds/c4/multilingual/3.0.1/
```

## Pre-train

With ``PRE=1``, it uses a [Preemptible TPU](https://cloud.google.com/tpu/docs/preemptible).

```bash
cp config.example.json config.mc4_mywiki40b.json
# Edit if needed
# "vocab_size" was 8064 when 8k sentencepiece

PROJECT=YOUR_PROJECT_NAME \
ZONE=YOUR_TPU_ZONE \
DATA_DIR=gs://YOUR_GSC_BUCKET/tfds \
SPM=gs://YOUR_GSC_BUCKET/spm/spm.wiki40b-ja.32k.byte_fallback.forT5.model \
OUT_DIR=gs://YOUR_GSC_BUCKET/out/mc4_mywiki40b \
TASK=mc4_mywiki40b \
CONFIG=config.mc4_mywiki40b.json
PRE=1 \
    poetry run bash -x tpu.sh
```

It needed 126 hours for 1M steps with TPU v3-8.
