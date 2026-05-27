from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # output size = vocab size
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output


LSTM_model = LSTMModel(
    vocab_size=3040,
    embedding_dim=128,
    hidden_dim=256
)
LSTM_model.load_state_dict(
    torch.load("lstm_next_word.pth", map_location=torch.device('cpu'), weights_only=False)
)

LSTM_model.eval()



with open('word_to_id.json', 'r') as f:
    word_to_id = json.load(f)

with open('id_to_word.json', 'r') as f:
    id_to_word = json.load(f)


def predict_next_words(text, k=3):
    text = text.lower()

    tokens = text.split()

    seq = [word_to_id[word] for word in tokens if word in word_to_id]

    seq = torch.tensor(seq, dtype=torch.long)

    pad_len = 19 - len(seq)

    seq = nn.functional.pad(seq, (pad_len, 0), value=0)

    seq = seq.unsqueeze(0)

    with torch.no_grad():

        pred = LSTM_model(seq)
        pred_prob = F.softmax(pred,dim=1)
        prob, indices = torch.topk(pred_prob, k=k, dim=1)
        

    indices_list = indices.squeeze(0).tolist()
    prob_list = prob.squeeze(0).tolist()

    words = [id_to_word[str(index)] for index in indices_list]

    return words,prob_list



app = FastAPI()

class Text(BaseModel):
    text: str=Field(example="I am", description="Input text for next word prediction")


@app.get("/")
async def root():
    return {"message": "Welcome to the Next Word Prediction API. Use the /predict/ endpoint to get predictions."}


@app.post("/predict/")
async def predict_next_word(text: Text):
    words,prob = predict_next_words(text.text)
    return {"predicted_words": words,"prob":prob}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)