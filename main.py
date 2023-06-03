import streamlit as st
import joblib
from PIL import Image

image = Image.open('assets/logo-myxl.jpg')
st.image(image, use_column_width=True)
st.title("ANALISIS SENTIMEN ULASAN APLIKASI My XL")
st.write("Author: Kharisma Intan Safitri, Vania, Rima")
st.write("Natural Language Processing Project for Final Exam")

# load model
model = joblib.load('model/modelSVM.pkl')
# load vectorize
vectorizer = joblib.load('model/vectorizer.pkl')

# inputan
ulasan = st.text_input('Masukkan ulasan')
button = st.button('Predict')

if button:
    # pembobotan menggunakan vectorize
    x_new = vectorizer.transform([ulasan])

    # predict menggunakan model
    predictions = model.predict(x_new)
    sentimen_class = ['Negatif', 'Netral', 'Positif']
    for i in predictions:
        st.write("Sentimen: ", sentimen_class[i])
