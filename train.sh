#!/bin/bash

BUCKET_NAME=mlops-data-bucket
JOB_NAME=bucket_1_epoch_0_workers_v7

#gcloud ai-platform jobs submit training ${JOB_NAME} \
#    --region=europe-west1 \
#    --master-image-uri=gcr.io/mlops-sentiment-analysis-gcp/mlops-sentiment-analysis-image:latest \
#    --scale-tier=CUSTOM \
#    --master-machine-type=n1-standard-8 \
#    --master-accelerator=type=nvidia-tesla-k80,count=1 \

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west3 \
    --master-image-uri=gcr.io/mlops-sentiment-analysis-gcp/mlops-sentiment-analysis-image:latest \