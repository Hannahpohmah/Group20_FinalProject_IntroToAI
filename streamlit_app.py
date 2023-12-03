import streamlit as st
import pandas as pd
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

# Load the LSTM model and other necessary preprocessing components
# Loading model architecture from JSON and weights from HDF5
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")

# Load the tokenizer from the saved file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 193  # Set your max_len here

# Function to preprocess the input text and make predictions
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len, padding="pre", truncating="pre")
    prediction = loaded_model.predict(sequence)
    return prediction

# Streamlit app layout
st.title('Fake/Real Text Detector')
user_input = st.text_area('Enter your text here:')
prediction_button = st.button('Predict')

if prediction_button:
    if user_input:
        prediction = predict_sentiment(user_input)
        threshold = 0.5  # Set the threshold for binary classification
        binary_prediction = (prediction > threshold).astype(int)

        # Convert binary prediction to label
        if binary_prediction[0][0] == 0:
            predicted_label = 'Fake'
        else:
            predicted_label = 'Real'

        st.write(f'Prediction: {predicted_label})')
    else:
        st.warning('Please enter some text for prediction.')

