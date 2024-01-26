import numpy as np
import pytest
from docclassifier.margin_based.infer import Inference, preprocess


def test_preprocess_not_numpy():
    with pytest.raises(ValueError):
        preprocess("not a numpy array")


def test_preprocess_resize():
    img = np.zeros((100, 100, 3))
    result = preprocess(img, img_size_infer=(50, 50), return_tensor=False)
    assert result['input']['img'].shape == (50, 50, 3)


def test_preprocess_return_tensor():
    img = np.zeros((100, 100, 3))
    result = preprocess(img, return_tensor=True)
    assert result['input']['img'].shape == (1, 3, 100, 100)


def test_inference_initialization():
    inference = Inference()
    assert inference is not None


def test_extract_feature():
    inference = Inference()
    img = np.random.rand(128, 128, 3)
    feature = inference.extract_feature(img)
    assert feature.shape == (256,)


def test_compare():
    inference = Inference()
    feat1 = np.array([1, 0, 0])
    feat2 = np.array([0, 1, 0])
    score = inference.compare(feat1, feat2)
    assert 0 <= score <= 1


def test_call_method():
    inference = Inference()
    img = np.random.rand(128, 128, 3)
    result = inference(img)
    assert isinstance(result, tuple)
    assert isinstance(result[0], (str, type(None)))
    assert isinstance(result[1], float)
