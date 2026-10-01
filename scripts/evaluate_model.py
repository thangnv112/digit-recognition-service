import gzip
import os
import urllib.request

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def download_mnist(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)
    data = data / 255.0
    data = (data - 0.1307) / 0.3081
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def main():
    model_path = 'weights/mnist_cnn.onnx'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train it first.")
        return

    os.makedirs('data/MNIST/raw', exist_ok=True)
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    images_url = base_url + 't10k-images-idx3-ubyte.gz'
    labels_url = base_url + 't10k-labels-idx1-ubyte.gz'
    
    images_file = 'data/MNIST/raw/t10k-images-idx3-ubyte.gz'
    labels_file = 'data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
    
    download_mnist(images_url, images_file)
    download_mnist(labels_url, labels_file)
    
    test_images = load_mnist_images(images_file)
    test_labels = load_mnist_labels(labels_file)

    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    batch_size = 128
    predictions = []
    
    print("Running inference...")
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        result = session.run(None, {input_name: batch})[0]
        preds = np.argmax(result, axis=1)
        predictions.extend(preds)
        
    predictions = np.array(predictions)
    
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))

if __name__ == "__main__":
    main()
