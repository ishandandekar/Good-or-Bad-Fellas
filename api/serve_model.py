from fastapi import FastAPI, HTTPException
import tensorflow as tf
import pickle
import numpy as np
import string
import re
import os
from pathlib import Path
import uvicorn

app = FastAPI(
    title="Good or Bad Fellas API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="69.420",
)

model = tf.keras.models.load_model("models/model_0_conv1d.h5")


def standardization(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "<br />", " ")
    return tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")


MAX_FEATURES = 20000
EMBED_DIM = 128
SEQ_LENGTH = 500


loaded_vectorizer = pickle.load(
    open("models/artifacts/model_0_conv1d_vectorization.pkl", "rb")
)
print(type(loaded_vectorizer), end="")
print("^^^^^^^^^^^^^^")
print(loaded_vectorizer.keys(), end="")
print("^^^^^^^^^^^^^^")
print([type(value) for value in loaded_vectorizer.values()], end="")
print("^^^^^^^^^^^^^^")
vectorizer = tf.keras.layers.TextVectorization(
    standardize=standardization,
    max_tokens=MAX_FEATURES,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
).set_weights(loaded_vectorizer["weights"])
print(type(vectorizer), end="")
print("^^^^^^^^^^^^^^")


def standardization(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "<br />", " ")
    return tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")


# def vectorize_text(text, vectorizer):
#     # text = tf.expand_dims(text, -1)
#     return vectorizer(text)


@app.get("/")
async def index():
    return {
        "Hello!": "This is an empty index page, head over to docs to see other API endpoints"
    }


@app.get("/predict-review/")
async def predict(review: str):
    try:
        text = tf.strings.lower(review)
        text = tf.strings.regex_replace(text, "<br />", " ")
        text = tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")
        print(f"{type(text)} ???????")
        print(f"{type(vectorizer)} #######")
        text = vectorizer(text)
        print(f"{type(text)} $$$$$$$$$")
        prediction = model.predict([text])
        print(prediction)
        return {"Prediction": prediction}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
