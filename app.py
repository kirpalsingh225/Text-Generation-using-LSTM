import streamlit as st
from keras.models import load_model
import numpy as np
import pickle
from keras.utils import pad_sequences
import time

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
model = load_model("lstm_text_gen.h5")
max_len = 195

st.title("Text Generation using LSTM")


user_input = st.text_input("Enter seed text:")


if st.button("Generate Text"):
    if user_input:
        text = user_input
        for i in range(100):
            token_text = tokenizer.texts_to_sequences([text])[0]
            padded = pad_sequences([token_text], maxlen=max_len-1, padding="pre")
            pos = np.argmax(model.predict(padded), axis=1)[0]
            for word, index in tokenizer.word_index.items():
                if index==pos:
                    text = text+" "+word
        st.write("Generated Text : ")
        st.write(text)
        
    else:
        st.warning("Please enter a seed text.")