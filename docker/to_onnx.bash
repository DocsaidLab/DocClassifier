#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/torch2onnx.py:/code/torch2onnx.py \
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_torch2onnx

if __name__ == \"__main__\":
    Fire(main_classifier_torch2onnx)
' > /code/torch2onnx.py && python /code/torch2onnx.py --cfg_name $1
"