import requests
import torch
from source.server.utils import model, emotions, transform
from PIL import Image

def predict_model_local(img):
    pil_image = Image.open(img)
    pil_image = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        output = model(pil_image)
        _, predicted = torch.max(output.data, 1)
        return emotions[predicted.item()]

def predict_model_api(image_path, image_file):
    files = {"file": (image_path, image_file, "image/jpg")}
    response = requests.post(url="http://localhost:8000/model/detect-emotion/", files=files)
    if response.status_code == 200:
        data = response.json()
        try:
            probabilites =  data["0"]["emotion_probabilities"]
            return max(probabilites, key=probabilites.get)
        except KeyError:
            print("Key not found in the response.")
    else:
        print(f"Request failed with status code: {response.status_code}")

def test_model(image_path, image_file): 
    model_api_prediction = predict_model_api(image_path, image_file)
    model_local_prediction = predict_model_local(image_file)
    if model_api_prediction is not None and model_local_prediction is not None: 
        assert model_local_prediction == model_api_prediction, "The api response and model response were different." 
        print("Everything is good !!!")
    else:
        print("The response of api or the response of the model is None.")

if __name__ == '__main__':
    img_path = '/home/hovhannes/Desktop/FacialExpressionRecognition/tests/functional_tests/images/happy_face.jpg' 
    with open(img_path, "rb") as image_file:
        test_model(image_path=img_path, image_file=image_file)