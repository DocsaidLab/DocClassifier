import numpy as np
import pytest
from docclassifier import DocClassifier, ModelType


def test_doc_classifier_initialization():
    classifier = DocClassifier()
    assert classifier is not None


def test_invalid_model_cfg():
    with pytest.raises(ValueError):
        DocClassifier(model_cfg="invalid_model_cfg")


def test_model_type_handling():
    for model_type in ModelType:
        classifier = DocClassifier(model_type=model_type)
        assert classifier is not None


def test_list_models():
    classifier = DocClassifier()
    models = classifier.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_call_method():
    classifier = DocClassifier()
    img = np.random.rand(128, 128, 3)
    result = classifier(img)
    assert isinstance(result, tuple)
    assert isinstance(result[0], (str, type(None)))
    assert isinstance(result[1], float)
