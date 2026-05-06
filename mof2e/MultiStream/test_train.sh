#!/bin/bash
# Test script for multi-stream model

cd /home/dell/autodl-tmp/lorafair/fairchem

python main.py \
    --mode train \
    --config-yml /home/dell/autodl-tmp/lorafair/fairchem/mof2e/MultiStream/MultiStream.yml \
    --run-dir runs/multistream_test
