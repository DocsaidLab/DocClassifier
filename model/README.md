# Dataset

In the training dataset, the data path has been set:

```python
INDOOR_ROOT = '/data/Dataset/indoor_scene_recognition/Images'
IMAGENET_ROOT = '/data/Dataset/ILSVRC2012/train'
```

In order to train the model, please download the dataset from the following websites and put them in the corresponding directories:

1. [Indoor Scene Recognition](https://web.mit.edu/torralba/www/indoor.html)
2. [ImageNet](http://www.image-net.org/)

If you want to put it in another directory, please modify `INDOOR_ROOT` and `IMAGENET_ROOT` in `dataset.py`.
