#!/bin/bash

BUCKET_NAME=mlops-data-bucket
JOB_NAME=job_4

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west3 \
    --master-image-uri=gcr.io/mlops-sentiment-analysis/mlops-sentiment-analysis-image:latest \