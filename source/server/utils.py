from PIL import Image

from image_processor import image_processing as ip
from model.net import CNN
import torch
import torch.optim as optim
from model.emotion_dataset import transform

emotions = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Creating model with random weights
model = CNN()
optimizer = optim.Adam(model.parameters())

# Load a model
checkpoint = torch.load('/home/hovhannes/Desktop/FacialExpressionRecognition/source/model/checkpoints/model.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epochs']
accuracy = checkpoint['accuracy']
loss = checkpoint['loss']


def get_face_coordinates_dict(coordinates):
    return {
        'x': str(coordinates[0]),
        'y': str(coordinates[1]),
        'w': str(coordinates[2]),
        'h': str(coordinates[3])
    }


def get_predicted_emotions(image, processor=None):
    model.eval()

    global emotions
    if processor is None:
        processor = ip.ImageProcessor(image)
        processor.detect_faces()

    faces = processor.get_cropped_faces()

    response = {}
    for face_id, face in faces.items():
        pil_face = Image.fromarray(face)
        img = transform(pil_face).unsqueeze(0)

        emotions_prob = {}
        with torch.no_grad():
            output = model(img)
            for i in range(len(output.data[0])):
                emotions_prob[emotions[i]] = round(output.data[0][i].item() * 100, 2)

        response[face_id] = {"emotion_probabilities": emotions_prob,
                             "face_coordinates": get_face_coordinates_dict(processor.faces[face_id])}
    return response


def get_labeled_image(image):
    model.eval()

    processor = ip.ImageProcessor(image)
    processor.detect_faces()

    emotions_response = get_predicted_emotions(image, processor)
    face_labels = {face_id: max(emotions_prob['emotion_probabilities'], key=emotions_prob['emotion_probabilities'].get)
                   for face_id, emotions_prob in emotions_response.items()}
    labeled_image = processor.add_labels(face_labels)

    return labeled_image