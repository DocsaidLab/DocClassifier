#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v /data/Dataset:/data/Dataset \
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"