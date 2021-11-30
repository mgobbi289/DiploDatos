#!/usr/bin/env bash

set -ex

if [ ! -d "./data/MeLi_Challenge/" ]
then
    mkdir -p ./data
    echo >&2 "Downloading MeLi Challenge Dataset"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/meli-challenge-2019.tar.bz2 -o ./data/MeLi_Challenge.tar.bz2
    tar jxvf ./data/MeLi_Challenge.tar.bz2 -C ./data/
fi

if [ ! -f "./data/SBW-vectors-300-min5.txt.gz" ]
then
    mkdir -p ./data
    echo >&2 "Downloading SBWCE"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/SBW-vectors-300-min5.txt.gz -o ./data/SBW-vectors-300-min5.txt.gz
fi

# Be sure the correct nvcc is in the path with the correct pytorch installation
export CUDA_HOME=/opt/cuda/10.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

python -m experiment.baseline_MLP \
    --train-data ./data/MeLi_Challenge/spanish.train.jsonl.gz \
    --validation-data ./data/MeLi_Challenge/spanish.validation.jsonl.gz \
    --test-data ./data/MeLi_Challenge/spanish.test.jsonl.gz \
    --token-to-index ./data/MeLi_Challenge/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish

