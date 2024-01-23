#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocClassifier.model import main_docclassifier_train

if __name__ == '__main__':
    Fire(main_docclassifier_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_classifier_train python trainer.py --cfg_name $1
