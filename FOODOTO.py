import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup


model = load_model('finalandlastmodelfood.h5')
labels = {0: 'apple', 1: 'banana', 2: 'cabbage', 3: 'carrot', 4: 'cauliflower', 5: 'grapes', 6: 'mango',
          7: 'orange', 8: 'potato', 9: 'tomato', 10: 'watermelon'}

fruits = ['Apple', 'Banana', 'Grapes', 'Mango', 'Orange', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Cabbage', 'Carrot', 'Cauli Flower', 'Potato', 'Tomato']

banner = st.image("logowel.png", use_column_width=True)


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("FOODOTO")
    # Write a styled paragraph of text
    st.markdown("<p style='font-size: 20px; color: green; font-style: italic;'>A food recognizer and calorie counting app is a useful tool for anyone who wants to track their daily caloric intake and maintain a healthy diet. This type of app uses image recognition technology to identify the food that you are eating, and then calculates the number of calories in that food based on its nutritional content.</p>",
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')


run()
