import base64
import os

import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "mnist_cnn.onnx")
MODEL_EXISTS = os.path.exists(MODEL_PATH)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model file not found")
def test_predict_valid_image(client, sample_image_bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.png", sample_image_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detected_text" in data
    assert "details" in data
    assert "inference_time_ms" in data
    assert len(data["details"]) > 0


def test_predict_invalid_file(client, invalid_bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", invalid_bytes, "text/plain")}
    )
    assert response.status_code == 400


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model file not found")
def test_predict_base64_endpoint(client, sample_image_bytes):
    b64_str = base64.b64encode(sample_image_bytes).decode("utf-8")
    response = client.post(
        "/api/v1/predict/base64",
        json={"image_base64": b64_str}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_predict_no_file(client):
    response = client.post("/api/v1/predict")
    assert response.status_code == 422
