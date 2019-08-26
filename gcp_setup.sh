#!/usr/bin/env bash

export BUCKET_NAME="zeusgan-mlengine"
export REGION="europe-west2"
export BUCKET_URI=gs://$BUCKET_NAME


if [[ ! $(gsutil ls -b $BUCKET_URI 2> /dev/null) ]]; then
    gsutil mb -l $REGION $BUCKET_URI
fi

#tar -czvf zeus_training.tar.gz training_imgs
export CLOUD_PATH_TRAINING=$BUCKET_URI/training_imgs
gsutil cp -r training_imgs ${CLOUD_PATH_TRAINING}