import numpy as np
import onnxruntime as ort
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DigitClassifier:
    def __init__(self, model_path: str):
        self.session = None
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            logger.info("ONNX model loaded successfully", model_path=model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model", model_path=model_path, error=str(e))

    def is_loaded(self) -> bool:
        return self.session is not None

    def predict(self, batch_tensor: np.ndarray) -> tuple[list[int], list[float]]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        outputs = self.session.run(None, {self.input_name: batch_tensor})[0]
        
        # Softmax with numerical stability trick
        max_logits = np.max(outputs, axis=1, keepdims=True)
        exp_logits = np.exp(outputs - max_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        return predictions.tolist(), confidences.tolist()
