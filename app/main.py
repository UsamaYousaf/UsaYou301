from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_class
from model.model import __version__ as model_version


app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict")
def predict(payload: TextIn):
    phrase_class = predict_class(payload.text)
    return {"phrase": phrase_class}
