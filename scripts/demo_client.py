"""End-to-End Demo Script for Handwritten Digit Recognition Microservice.

This script:
1. Generates a realistic test image containing a sequence of handwritten digits.
2. Sends the image to the FastAPI microservice endpoints (/health, /predict, /predict/base64).
3. Prints the structured response, detected digits, confidences, and inference time.
4. Generates an annotated image showing bounding boxes and predictions.
"""

import os
import sys
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app


def create_demo_digit_image(text="804", save_path="demo_input.png"):
    """Create a realistic demo image with handwritten-style digits on white paper."""
    h, w = 120, 300
    img = np.ones((h, w), dtype=np.uint8) * 255

    # Draw digits with slight offsets and handwriting-like stroke width
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_x = 40
    step_x = 75

    for i, ch in enumerate(text):
        x = start_x + i * step_x + np.random.randint(-3, 4)
        y = 80 + np.random.randint(-4, 5)
        scale = 1.8 + np.random.uniform(-0.1, 0.1)
        thickness = 4 + np.random.randint(0, 2)
        cv2.putText(img, ch, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

    # Add slight paper texture / subtle noise
    noise = np.random.normal(0, 3, (h, w)).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, noisy_img)
    print(f"Created demo test image: {save_path}")
    return noisy_img, save_path


def run_demo():
    print("=" * 60)
    print("   HANDWRITTEN DIGIT RECOGNITION MICROSERVICE DEMO")
    print("=" * 60)

    # 1. Create client and test health endpoint
    print("\n[Step 1] Testing GET /health...")
    with TestClient(app) as client:
        health_resp = client.get("/health")
        print(f"Status Code: {health_resp.status_code}")
        print(f"Response: {health_resp.json()}")
        assert health_resp.status_code == 200

        # 2. Create demo image with digits "804"
        print("\n[Step 2] Generating sample handwritten image '804'...")
        input_image_path = os.path.join(os.path.dirname(__file__), "..", "demo_input.png")
        img, img_path = create_demo_digit_image(text="804", save_path=input_image_path)

        # 3. Test Multipart File Upload /api/v1/predict
        print("\n[Step 3] Testing POST /api/v1/predict (Multipart Form)...")
        with open(input_image_path, "rb") as f:
            file_bytes = f.read()

        predict_resp = client.post(
            "/api/v1/predict",
            files={"file": ("demo_input.png", file_bytes, "image/png")}
        )

        print(f"Status Code: {predict_resp.status_code}")
        result = predict_resp.json()
        print(f"Detected Text:     '{result.get('detected_text')}'")
        print(f"Inference Time:    {result.get('inference_time_ms', 0):.2f} ms")
        print("Details:")
        for idx, item in enumerate(result.get("details", []), 1):
            digit = item["digit"]
            conf = item["confidence"] * 100
            box = item["box"]
            print(f"  Digit #{idx}: Value = {digit} | Confidence = {conf:.2f}% | Box [x, y, w, h] = {box}")

        # 4. Test Base64 Endpoint /api/v1/predict/base64
        print("\n[Step 4] Testing POST /api/v1/predict/base64 (JSON Payload)...")
        b64_str = base64.b64encode(file_bytes).decode("utf-8")
        b64_resp = client.post(
            "/api/v1/predict/base64",
            json={"image_base64": b64_str}
        )
        print(f"Status Code: {b64_resp.status_code}")
        b64_result = b64_resp.json()
        print(f"Base64 Detected Text: '{b64_result.get('detected_text')}'")

        # 5. Test Prometheus Metrics /metrics
        print("\n[Step 5] Testing GET /metrics (Prometheus)...")
        metrics_resp = client.get("/metrics")
        print(f"Status Code: {metrics_resp.status_code}")
        sample_metrics = [line for line in metrics_resp.text.splitlines() if line.startswith("request_count") or line.startswith("request_latency")]
        print("Sample Metrics recorded:")
        for m in sample_metrics[:4]:
            print(f"  {m}")

        # 6. Annotate image with predictions
        print("\n[Step 6] Saving annotated prediction visualization...")
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for item in result.get("details", []):
            x, y, w, h = item["box"]
            digit = item["digit"]
            conf = item["confidence"]
            # Draw green bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            # Draw label background and text
            label = f"{digit} ({conf*100:.1f}%)"
            cv2.putText(annotated, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

        output_path = os.path.join(os.path.dirname(__file__), "..", "demo_output_annotated.png")
        cv2.imwrite(output_path, annotated)
        print(f"Annotated result saved to: {output_path}")

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED SUCCESSFULLY! ALL SYSTEMS FUNCTIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
