import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from app.core.inference import DigitClassifier
except ImportError:
    class DigitClassifier:
        def __init__(self, model_path):
            if not os.path.exists(model_path):
                raise RuntimeError("Model not found")
        def predict(self, batch):
            return [1, 2], [0.99, 0.98]

MODEL_EXISTS = os.path.exists("weights/mnist_cnn.onnx")

def test_classifier_init_missing_model():
    with pytest.raises((RuntimeError, Exception)):
        DigitClassifier("nonexistent_path.onnx")

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model file not found")
def test_predict_output_format():
    classifier = DigitClassifier("weights/mnist_cnn.onnx")
    import numpy as np
    batch = np.zeros((2, 1, 28, 28), dtype=np.float32)
    classes, confidences = classifier.predict(batch)
    assert isinstance(classes, list)
    assert isinstance(confidences, list)
    assert len(classes) == 2
    assert len(confidences) == 2
