import requests

def test_api(image_path, image_file): 
    files = {"file": (image_path, image_file, "image/jpg")}
    response = requests.post(url="http://localhost:8000/model/detect-emotion/", files=files)
    if response.status_code == 200:
        data = response.json()
        assert data == {}, "The api detect faces on nature image."
        print("Everything is good !!!")
    else:
        print(f"Request failed with status code: {response.status_code}")

if __name__ == '__main__':
    image_path = '/home/hovhannes/Desktop/FacialExpressionRecognition/tests/functional_tests/images/nature.jpg' 
    with open(image_path, "rb") as image_file:
        test_api(image_path, image_file)