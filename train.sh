#!/bin/bash

BUCKET_NAME=bucket-mlops-project-1
JOB_NAME=job_8
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west3 \
    --master-image-uri=gcr.io/mlops-sentiment-analysis-gcp/mlops-sentiment-analysis-image:latest \
    --job-dir=${JOB_DIR} \
