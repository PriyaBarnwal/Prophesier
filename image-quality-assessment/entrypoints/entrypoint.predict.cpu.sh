#!/bin/bash
set -e

BASE_MODEL_NAME=$1

# predict
python -m evaluater.predict_csv \
--base-model-name $BASE_MODEL_NAME \
--csv-file /src/dataset.csv
