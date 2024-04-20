import streamlit as st
import tensorflow as tf
import numpy as np
from classes import class_names

st.header("Food Vision")
st.write("Classify the food item from the given image")



with st.form("file_input_form"):
    image = st.file_uploader("Upload the image")
    submitted = st.form_submit_button("Submit")

def preprocess_image(img):
    file_bytes = img.read()
    file_tensor = tf.image.decode_image(file_bytes,channels=3)
    img = tf.image.resize(file_tensor, size=(224,224))
    img = tf.expand_dims(img, axis=0)
    return img
if submitted:
    model_input = tf.keras.layers.Input(shape=(224, 224, 3))
    tfsmlayer = tf.keras.layers.TFSMLayer("food_vision_101_classes/tuned_model", call_endpoint='serving_default')(model_input)
    model = tf.keras.Model(inputs=model_input, outputs=tfsmlayer)
    processed_image = preprocess_image(image)
    pred_probs = model.predict(processed_image).get('softmax_float32')
    pred_index = pred_probs.argmax(axis=1)
    st.image(image, width=200)
    st.write(f"## {class_names[pred_index[0]]}")