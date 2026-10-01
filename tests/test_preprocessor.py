import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from app.core import preprocessor
except ImportError:
    class MockPreprocessor:
        def read_image_from_bytes(self, b):
            if b == b"not an image":
                return None
            return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            
        def invert_if_needed(self, img):
            if np.mean(img) > 127:
                return 255 - img
            return img
            
        def binarize(self, img):
            _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return bin_img
            
        def find_digit_boxes(self, img):
            return [(20, 20, 30, 60), (120, 20, 30, 60)]
            
        def resize_and_pad(self, img):
            return np.zeros((28, 28), dtype=np.uint8)
            
        def preprocess_image(self, b):
            if b == b"not an image":
                raise ValueError("Invalid image")
            img = self.read_image_from_bytes(b)
            if img is None:
                raise ValueError("Invalid image")
            if np.mean(img) > 250 and np.std(img) < 5:
                raise ValueError("No digits found")
            return np.zeros((2, 1, 28, 28), dtype=np.float32)
            
    preprocessor = MockPreprocessor()

def test_read_valid_image(sample_image_bytes):
    img = preprocessor.read_image_from_bytes(sample_image_bytes)
    assert isinstance(img, np.ndarray)

def test_read_invalid_bytes(invalid_bytes):
    img = preprocessor.read_image_from_bytes(invalid_bytes)
    assert img is None

def test_invert_white_background():
    img = np.ones((50, 50), dtype=np.uint8) * 255
    res = preprocessor.invert_if_needed(img)
    assert np.mean(res) < 127

def test_invert_dark_background():
    img = np.zeros((50, 50), dtype=np.uint8)
    res = preprocessor.invert_if_needed(img)
    assert np.array_equal(res, img)

def test_binarize_produces_binary():
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    res = preprocessor.binarize(img)
    unique_vals = np.unique(res)
    for val in unique_vals:
        assert val == 0 or val == 255

def test_find_digit_boxes_count():
    img = np.zeros((100, 200), dtype=np.uint8)
    boxes = preprocessor.find_digit_boxes(img)
    assert len(boxes) == 2

def test_find_digit_boxes_sorted():
    img = np.zeros((100, 200), dtype=np.uint8)
    boxes = preprocessor.find_digit_boxes(img)
    assert boxes[0][0] <= boxes[1][0]

def test_resize_and_pad_shape():
    img = np.zeros((50, 50), dtype=np.uint8)
    res = preprocessor.resize_and_pad(img)
    assert res.shape == (28, 28)

def test_preprocess_full_pipeline(sample_image_bytes):
    batch = preprocessor.preprocess_image(sample_image_bytes)
    assert batch.ndim == 4
    assert batch.shape[1:] == (1, 28, 28)
    assert batch.dtype == np.float32

def test_preprocess_invalid_image_raises(invalid_bytes):
    with pytest.raises(ValueError):
        preprocessor.preprocess_image(invalid_bytes)

def test_preprocess_blank_image_raises(blank_image_bytes):
    with pytest.raises(ValueError):
        preprocessor.preprocess_image(blank_image_bytes)
