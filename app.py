import streamlit as st
import tensorflow as tf
import numpy as np
from classes import class_names
import tensorflow_hub as hub
from tensorflow.keras import mixed_precision

# Define your custom layer
class RandomHeight(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomHeight, self).__init__()

    def call(self, inputs):
        # Your layer logic here
        return inputs

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
    # Wrap your code that loads the model with custom_object_scope
    mixed_precision.set_global_policy('mixed_bfloat16')
    model = tf.keras.models.load_model("food_vision_101_classes.keras", compile=False)
    
    processed_image = preprocess_image(image)
    pred_probs = model.predict(processed_image)
    # print(pred_probs)
    pred_index = np.argmax(pred_probs, axis=1)
    st.image(image, width=200)
    st.write(f"## {class_names[pred_index[0]]}")
