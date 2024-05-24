import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.responses import JSONResponse
import io
from server.utils import *

app = FastAPI()


@app.get("/model/summary",
         summary="Get Model Summary",
         tags=["Model Information"])
def get_model():
    """
    Retrieve the summary or evaluation details of the model.

    Returns:
    - **Dict[str, str]**: A dictionary containing the model's summary or evaluation details.
    """
    return {
        'name': 'Facial Emotion Recognizer',
        'epochs': epochs,
        'accuracy': round(accuracy, 2),
        'loss': round(loss, 2)
    }


@app.get("/model/emotions",
         tags=["Model Information"])
def get_emotions():
    """
    This endpoint returns a list of supported emotions that the model can recognize.
    """
    return {"supported_emotions": emotions}


@app.get("/model/emotions/{emotion_id}",
         summary="Get Emotion by id",
         tags=["Model Information"])
def get_emotion_by_id(emotion_id: int):
    """
    Retrieve an emotion based on the provided emotion ID.

    Args:
    - **emotion_id** (int): The ID of the emotion to retrieve. Must be between 0 and 6 (inclusive).

    Raises:
    - **HTTPException**: If the emotion_id is not in the range 0 to 6, a 400 status code is returned with an appropriate
    error message.

    Returns:
    - **Dict[str, str]**: A dictionary containing the emotion corresponding to the provided ID.
    """
    if emotion_id < 0 or emotion_id > 6:
        raise HTTPException(status_code=400, detail="Id must be in the range(0, 6)")
    return {"emotion": emotions[emotion_id]}


@app.post("/model/detect-emotion/",
          summary="Get face emotions",
          tags=["Model Prediction"])
async def detect_emotion(
        file: UploadFile = File(description="A required image to detect face emotion")
):
    """
    Receives an image file and detects all faces from the image, then faces emotions using a predefined model.

    Args:
    - **file** (UploadFile): The image file to detect emotions. Must be in a valid image format.

    Returns:
    - **dict**: The dictionary where the key is the face ID detected from the image and the value
    is the highest probability predicted emotion by the model.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    return get_predicted_emotions(image)


@app.post("/model/detect-emotion/image",
          summary="Get face emotions",
          tags=["Model Prediction"])
async def label_image(
        file: UploadFile = File(description="A required image to detect face emotion")
):
    """
    Receives an image file and detects all face emotions using a predefined model, returning the segmented
    image as a base64-encoded PNG string.

    Args:
    - **file** (UploadFile): The image file to detect emotions. Must be in a valid image format.

    Returns:
    - **dict**: The dictionary where the key is the face ID detected from the image and the value
    is the highest probability predicted emotion by the model.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    labeled_image = get_labeled_image(image)

    _, encoded_image = cv2.imencode('.jpg', labeled_image)
    byte_image = encoded_image.tobytes()

    return Response(content=byte_image, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("controllers:app", host="0.0.0.0", port=8000, reload=True)
