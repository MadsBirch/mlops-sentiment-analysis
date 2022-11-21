#!/bin/bash

BUCKET_NAME=mlops-data-bucket
JOB_NAME=job_1
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west3 \
    --master-image-uri=gcr.io/mlops-sentiment-analysis-gcp/mlops-sentiment-analysis-image:latest \
    --job-dir=${JOB_DIR} \


gcloud ai-platform jobs submit training ${JOB_NAME} \
  --region=us-central1 \
  --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-10 \
  --scale-tier=CUSTOM \
  --master-machine-type=n1-standard-8 \
  --master-accelerator=type=nvidia-tesla-p100,count=1 \
  --job-dir=${JOB_DIR} \
  --package-path=./trainer \
  --module-name=trainer.task \
  -- \
  --train-files=gs://cloud-samples-data/ai-platform/chicago_taxi/training/small/taxi_trips_train.csv \
  --eval-files=gs://cloud-samples-data/ai-platform/chicago_taxi/training/small/taxi_trips_eval.csv \
  --num-epochs=10 \
  --batch-size=100 \
  --learning-rate=0.001