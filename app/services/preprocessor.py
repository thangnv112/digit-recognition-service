import cv2
import numpy as np
from app.utils.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray | None:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

def invert_if_needed(gray: np.ndarray) -> np.ndarray:
    if np.mean(gray) > 127:
        return cv2.bitwise_not(gray)
    return gray

def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixel_ratio = np.sum(binary == 255) / binary.size
    
    if white_pixel_ratio < 0.01:
        logger.info("Otsu binarization failed, falling back to adaptive thresholding")
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        logger.info("Used Otsu binarization")
        
    return binary

def find_digit_boxes(binary: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
            
    boxes.sort(key=lambda b: b[0])
    return boxes

def resize_and_pad(digit_img: np.ndarray, target_size: int = 20, canvas_size: int = 28) -> np.ndarray:
    h, w = digit_img.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - new_h) // 2
    x_offset = (canvas_size - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    gray = read_image_from_bytes(image_bytes)
    if gray is None:
        raise ValueError("Invalid image")
        
    gray = invert_if_needed(gray)
    binary = binarize(gray)
    
    boxes = find_digit_boxes(binary, settings.min_contour_area)
    if not boxes:
        raise ValueError("No digits found")
        
    logger.info("Found digits", count=len(boxes))
    
    processed_digits = []
    for x, y, w, h in boxes:
        digit_img = binary[y:y+h, x:x+w]
        padded = resize_and_pad(digit_img)
        # Standardize using MNIST dataset mean and std used during training
        normalized = ((padded.astype(np.float32) / 255.0) - 0.1307) / 0.3081
        processed_digits.append(normalized[np.newaxis, ...])
        
    # Stack to [N, 1, 28, 28]
    batch_tensor = np.stack(processed_digits).astype(np.float32)
    return batch_tensor, boxes
