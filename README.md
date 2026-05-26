# Next Word Prediction using LSTM 🚀

A deep learning project for **Next Word Prediction** using an **LSTM model** built with **PyTorch**.
The project includes a **FastAPI backend** and a **Streamlit frontend** for interactive text generation.

The model predicts the next word from user input and can also generate the next **N words** sequentially.

---

## Features ✨

* Next word prediction using LSTM
* Streamlit frontend + FastAPI backend
* Dockerized deployment
* Vocabulary size: **3,037 words**
* Sequence length: **20**
* LSTM Accuracy: **66%**
* TCN model comparison included

---

## Tech Stack 🛠️

* Python
* PyTorch
* FastAPI
* Streamlit

---

## Dataset 📚

[Kaggle Dataset - Next Word Predictor NLP Task](https://www.kaggle.com/datasets/shorya22/next-word-predictor-dataset-nlp-task?utm_source=chatgpt.com)

---

## Project Workflow ⚙️

* **preprocess.ipynb** → Cleans dataset and creates `text` & `next_word` columns
* **EDA.ipynb** → Tokenization, padding, word-to-id mapping, creates `final_data.csv`
* **lstm.ipynb** → Trains and saves LSTM model
* **TCN.ipynb** → Trains and saves TCN model
* **testing_models.ipynb** → Compares LSTM and TCN performance
* **api.py** → FastAPI backend for predictions
* **app.py** → Streamlit frontend for user interaction

---

## Model Performance 📊

| Model | Accuracy |
| ----- | -------- |
| LSTM  | 66%      |
| TCN   | 36%      |


---
NOTE this can be improved with more data more traing 
---
## Author 👨‍💻

Kumar Ayush
