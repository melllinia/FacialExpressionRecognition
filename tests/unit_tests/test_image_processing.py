import pytest
import cv2
import numpy as np
from PIL import Image
from image_processor.image_processing import ImageProcessor

# Mock class to override cv2.CascadeClassifier
class MockCascadeClassifier:
    def __init__(self, *args, **kwargs):
        pass
    
    def detectMultiScale(self, *args, **kwargs):
        return np.array([[30, 30, 40, 40]])


def create_dummy_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rectangle =  cv2.rectangle(img, (30, 30), (70, 70), (255, 255, 255), -1)
    return rectangle

# Convert image to bytes
def image_to_bytes(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def test_load_image_from_path(monkeypatch):
    dummy_image = create_dummy_image()
    
    def mock_imread(path):
        return dummy_image

    monkeypatch.setattr(cv2, 'imread', mock_imread)
    processor = ImageProcessor("dummy_path.jpg")
    assert processor.img is not None
    assert processor.img.shape == (100, 100, 3)


def test_load_image_from_pillow():
    pil_image = Image.fromarray(create_dummy_image())
    processor = ImageProcessor(pil_image)
    assert processor.img is not None
    assert processor.img.shape == (100, 100, 3)


def test_load_image_from_bytes():
    dummy_image = create_dummy_image()
    image_bytes = image_to_bytes(dummy_image)
    processor = ImageProcessor(image_bytes)
    assert processor.img is not None
    assert processor.img.shape == (100, 100, 3)


def test_load_image_invalid_type():
    with pytest.raises(ValueError, match="Unsupported image input type."):
        ImageProcessor(12345)


def test_load_image_invalid_path(monkeypatch):
    def mock_imread(path):
        return None

    monkeypatch.setattr(cv2, 'imread', mock_imread)
    with pytest.raises(ValueError, match="Could not load the image."):
        ImageProcessor("invalid_path.jpg")


def test_detect_faces(monkeypatch):
    dummy_image = create_dummy_image()
    
    pil_image = Image.fromarray(dummy_image, 'RGB')
    processor = ImageProcessor(pil_image)

    monkeypatch.setattr(cv2, 'CascadeClassifier', MockCascadeClassifier)
    processor.detect_faces()
    assert processor.faces is not None
    assert len(processor.faces) == 1
    assert processor.faces[0] == [30, 30, 40, 40]


def test_get_cropped_faces(monkeypatch):
    dummy_image = create_dummy_image()
    
    pil_image = Image.fromarray(dummy_image, 'RGB')
    processor = ImageProcessor(pil_image)

    monkeypatch.setattr(cv2, 'CascadeClassifier', MockCascadeClassifier)
    processor.detect_faces()
    cropped_faces = processor.get_cropped_faces()
    assert cropped_faces is not None
    assert len(cropped_faces) == 1
    face = cropped_faces[0]
    assert face.shape == (40, 40, 3)


def test_add_labels(monkeypatch):
    dummy_image = create_dummy_image()
    
    pil_image = Image.fromarray(dummy_image, 'RGB')
    processor = ImageProcessor(pil_image)

    monkeypatch.setattr(cv2, 'CascadeClassifier', MockCascadeClassifier)
    processor.detect_faces()
    labeled_image = processor.add_labels(["Face 1"])
    assert labeled_image is not None
    assert processor.faces is not None
    assert processor.faces[0] == [30, 30, 40, 40]