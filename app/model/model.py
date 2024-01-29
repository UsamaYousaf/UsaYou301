from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the Keras model
model = load_model(f"{BASE_DIR}/phrase_model-{__version__}.keras")

# Load the tokenizer
with open(f"{BASE_DIR}/tokenizer-{__version__}.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the maximum length of sequences based on your model training
maxlen = 111

classes = ["ft", "mr", "ct", "pkg", "ch", "cnc"]


def preprocess_text(text):
    # Preprocessing steps as used in training
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence


def predict_class(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    return decode_prediction(prediction)


def decode_prediction(prediction):
    return classes[np.argmax(prediction)]
