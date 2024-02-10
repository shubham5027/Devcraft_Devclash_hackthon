hide_st_style = """
            <style>
            #MainMenu {visibility:hidden;}
            footer {visibility:hidden;}
            header {visibility:hidden;}
            </style>
            """
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64
import cv2
import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

header_image_path = 'farmi.jpg'
st.image(header_image_path, use_column_width='auto')


def get_gemini_repsonse(input,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([input,prompt])
    return response.text

input_prompt= """You are an farming expert and i want some remedial and preventive information about given tomato plant disease. give me remedial informaion for appropriate environmental condition , soil condition and what pesticides and fertilizers to use. give the information in such away that it is easy for a farmer to understand  if possible in hindi"""

MODEL = tf.keras.models.load_model('./potato_trained_models/1/')
TOMATO_MODEL = tf.keras.models.load_model('./tomato_trained_models/1')
PEEPER_MODEL = tf.keras.models.load_model('./pepper_trained_models/1')

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

Tomato_classes = ['Tomato_healthy', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato_Septoria_leaf_spot',
 'Tomato__Tomato_mosaic_virus', 'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot', 'Tomato_Late_blight',
 'Tomato_Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

pepper_classes = ['pepper_bell_bacterial_spot','pepper_healthy']


st.title("Plant Disease Detection")
st.write("This application is detecting disease in three plants photato, tomato and pepper")
options = ["Select One Plant","Tomato", "Potato", "Pepper"]


selected_option = st.selectbox("Select Plant:", options)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def read_file_as_image(data)->np.array:
    image = np.array(data)
    image = cv2.resize(image, (256,256))
    return image

async def potato():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        image = read_file_as_image(image)
        image_batch = np.expand_dims(image, axis=0)
        predictions = MODEL.predict(image_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print("prediction", class_names[np.argmax(predictions)])
        st.write("Predicted Class : ", predicted_class, " Confidence Level : ", confidence)
        input=st.text_input(predicted_class,key="input")
        response=get_gemini_repsonse(input_prompt,input)
        st.subheader("The Response is")
        st.write(response)

async def tomato():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        image = read_file_as_image(image)
        image_batch = np.expand_dims(image, axis=0)
        predictions = TOMATO_MODEL.predict(image_batch)
        predicted_class = Tomato_classes[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print("prediction", Tomato_classes[np.argmax(predictions)])
        st.write("Predicted Class : ", predicted_class, " Confidence Level : ", confidence)
        input=st.text_input(predicted_class,key="input")
        response=get_gemini_repsonse(input_prompt,input)
        st.subheader("The Response is")
        st.write(response)


async def pepper():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        image = read_file_as_image(image)
        image_batch = np.expand_dims(image, axis=0)
        predictions = PEEPER_MODEL.predict(image_batch)
        predicted_class = pepper_classes[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print("prediction", pepper_classes[np.argmax(predictions)])
        st.write("Predicted Class : ", predicted_class, "Confidence Level : ", confidence)
        input=st.text_input(predicted_class,key="input")
        response=get_gemini_repsonse(input_prompt,input)
        st.subheader("The Response is")
        st.write(response)



import asyncio

if __name__ == "__main__":
     if st.button('Predict'):
    
        if selected_option == 'Potato':
            asyncio.run(potato())
        elif selected_option == 'Tomato':
            asyncio.run(tomato())
        else :
            asyncio.run(pepper())
        # else:
        #     st.write("not avalible")
