import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_directory = os.path.dirname(os.path.abspath(__file__))
model_path = "{0}/trained_model/plant_disease_prediction_image_classifier.h5".format(working_directory)
model = tf.keras.models.load_model(model_path)

#loading the class names to class_indices.json file
class_indices = json.load(open("{0}/class_indices.json".format(working_directory)))

img_width = 224
img_height = 224
#Function to load and preprocess image
def load_and_preprocess_image(image_path):
    #Open and Resize image
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))

    #Convert image to NumPy array
    img_array = np.array(img)

    #Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    #Normalize pixel values(float data type)
    img_array = img_array.astype('float32') / 255.0

    return img_array

#Function to predict class of an image
def predict_class(model, image_path, class_indices):
    #Load and preprocess image
    preprocessed_img = load_and_preprocess_image(image_path)

    #Make prediction
    prediction = model.predict(preprocessed_img)

    #Get class with highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    #Get class label
    predicted_class_label = class_indices[str(predicted_class_index)]

    return predicted_class_label

#Streamlit App
st.title('Plant Disease Classification System')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    column1, column2 = st.columns(2)

    with column1:
        resized_image = image.resize((160, 160))
        st.image(resized_image)

    with column2:
        if st.button('Identify Disease'):
            prediction = predict_class(model, uploaded_image, class_indices)
            st.success("Predicted disease: {0}".format(str(prediction)))


