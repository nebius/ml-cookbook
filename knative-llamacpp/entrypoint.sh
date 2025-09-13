#!/bin/sh
aws s3 cp $S3_BUCKET/$MODEL_KEY /mnt/models/$MODEL_KEY
/app/llama-server