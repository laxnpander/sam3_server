#!/bin/bash

docker run -it \
  --gpus=all \
  --net=host \
  sam3_server:v1