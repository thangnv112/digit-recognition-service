import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure the demo project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """Create a synthetic image with 2 black rectangles on white background."""
    img = np.ones((100, 200), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (50, 80), 0, -1)
    cv2.rectangle(img, (120, 20), (150, 80), 0, -1)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture
def blank_image_bytes():
    """All white image — no digits to detect."""
    img = np.ones((100, 100), dtype=np.uint8) * 255
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture
def invalid_bytes():
    return b"not an image at all"
