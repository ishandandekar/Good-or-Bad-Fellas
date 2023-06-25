import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
from pydantic import BaseModel
import pickle
from PIL import Image
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("../models/model_0_conv1d.h5")
loaded_vectorizer = pickle.load(
    open("../models/artifacts/model_0_conv1d_vectorization.pkl")
)
vectorizer = (
    tf.keras.layers.TextVecorization()
    .from_config(loaded_vectorizer["config"])
    .set_weights(loaded_vectorizer["weights"])
)


class UserInput(BaseModel):
    user_input: float


@app.get("/")
async def index():
    return {
        "Hello!": "This is an empty index page, head over to docs to see other API endpoints"
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
    except:
        raise HTTPException(status_code=404, detail=f"File is not an image")

    prediction = model.predict([UserInput.user_input])
    return {"Prediction": tf.math.sigmoid(prediction)}
