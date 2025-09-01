#!/bin/sh

set -e  # exit on failure

CUR=$(pwd)
IMAGE_NAME=eqphases
CONTAINER_NAME=detector

echo "Building image: $IMAGE_NAME in $CUR"

# Build the image (tagging as latest explicitly)
docker build -t $IMAGE_NAME:latest .

# Confirm image exists
docker images | grep $IMAGE_NAME

# Remove old container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    docker rm -f $CONTAINER_NAME
fi

# Run interactively, mounting current dir, and Start the termninal
docker run --name $CONTAINER_NAME -it -v "$CUR":/eqphases \
    -v /eqphases/.venv -w /eqphases $IMAGE_NAME:latest bash 
# Ignore local .venv and set work directory to /eqphases
