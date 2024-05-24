import pytest
import requests

@pytest.fixture
def image_file():
    img_path = '/home/hovhannes/Desktop/FacialExpressionRecognition/tests/functional_tests/images/nature.jpg'
    with open(img_path, "rb") as file:
        yield file

def test_api(image_file): 
    files = {"file": ("happy_face.jpg", image_file, "image/jpg")}
    response = requests.post(url="http://localhost:8000/model/detect-emotion/", files=files)
    if response.status_code == 200:
        data = response.json()
        assert data == {}, "The API detected faces on the nature image."
    else:
        pytest.fail(f"Request failed with status code: {response.status_code}")