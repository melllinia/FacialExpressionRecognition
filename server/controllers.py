import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from model.net import CNN
import torch.optim as optim

app = FastAPI()

emotions = {
    0 : 'Angry',
    1 : 'Disgust',
    2 : 'Fear',
    3 : 'Happy',
    4 : 'Neutral',
    5 : 'Sad',
    6 : 'Surprise'
}

model = CNN()
optimizer = optim.Adam(model.parameters())

# Load a model
checkpoint = torch.load('/home/hovhannes/Desktop/FacialExpressionRecognition/model/checkpoints/model.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']

# Save this in train stage
# accuracy_on_validation_set = evaluate(model, val_loader)
# accuracy_on_train_set = evaluate(model, train_loader)

@app.get("/model/summary", 
         summary="Get Model Summary",
         tags = ["Model Information"])
def get_model():
    """
    Retrieve the summary or evaluation details of the model.

    Returns:
    - **Dict[str, str]**: A dictionary containing the model's summary or evaluation details.
    """
    return {"model" : model.eval()}


@app.get("/model/emotions", 
         tags = ["Model Information"])
def get_emotions():
    """
    This endpoint returns a list of supported emotions that the model can recognize.
    """
    return {"supported_emotions": emotions}


@app.get("/model/emotions/{emotion_id}", 
         summary="Get Emotion by id",
         tags = ["Model Information"])
def get_emotion_by_id(emotion_id : int):
    """
    Retrieve an emotion based on the provided emotion ID.

    Args:
    - **emotion_id** (int): The ID of the emotion to retrieve. Must be between 0 and 6 (inclusive).

    Raises:
    - **HTTPException**: If the emotion_id is not in the range 0 to 6, a 400 status code is returned with an appropriate error message.

    Returns:
    - **Dict[str, str]**: A dictionary containing the emotion corresponding to the provided ID.
    """
    if emotion_id < 0 or emotion_id > 6:
        raise HTTPException(status_code=400, detail="Id must be in the range(0, 6)")
    return {"emotion": emotions[emotion_id]} 

if __name__ == "__main__":
    uvicorn.run("controllers:app", host="0.0.0.0", port=8000, reload=True)
