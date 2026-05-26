import streamlit as st
import requests
st.title("Next Word Prediction")
st.write("Enter a sequence of words to predict the next word.")
st.sidebar.write(f"This is a Streamlit app that interacts with a FastAPI backend to predict the next word based on user input. The backend uses a pre-trained LSTM model with 66% accuracy to generate predictions. It was trained on a vocabulary of 3,037 words with a sequence length of 20 words.")

with st.form("prediction_form"):
    user_input = st.text_input("Input Text", placeholder="Type a sequence of words...")
    next_n_words = st.number_input("Numbers of next words",min_value=1,max_value=5)
    submit_button = st.form_submit_button("Predict")

if submit_button:
    if user_input.strip() == "":
        st.warning("Please enter some text to predict the next word.")
    else:
        response = requests.post("http://localhost:8000/predict/", json={"text": user_input})
        if response.status_code == 200:
            predicted_words = response.json().get("predicted_words", [])
            st.write("Predicted Next Words:")
            for word in predicted_words:
                text = user_input+ " " + word
                for i in range(1,next_n_words):
                    response = requests.post("http://localhost:8000/predict/", json={"text": text})
                    text = text + " " + response.json().get("predicted_words", [])[0]
                st.write(text)
        else:
            st.error("Error occurred while predicting. Please try again.")