import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
import pickle
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import text_processing_pipeline

# ----- CONFIG (TRAINING) ----------------
MAX_WORDS = 20000
MAX_LEN = 80
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "LSTMModel/checkpoints/lstm_checkpoint.pth"
# CHECKPOINT_PATH = "LSTMMODEL/lstm_model_final.pth"
TOKENIZER_PATH = "LSTMModel/models/tokenizer.pkl"
LABEL_ENCODER_PATH = "LSTMModel/models/label_encoder.pkl"

# ----- MODEL (EXACT SAME AS TRAINING) ------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def load_lstm_model():
    model = LSTMClassifier(
        vocab_size=MAX_WORDS,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_SIZE,
        output_dim=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

model = load_lstm_model()

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

def prepare_input(texts):
    processed_texts = [text_processing_pipeline(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(processed_texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post")

def view_emotions_lstm(comments):
    X = prepare_input(comments)
    X = torch.tensor(X, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        outputs = model(X)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    # map integers to original emotion labels using label_encoder
    labels = label_encoder.inverse_transform(preds)

    counts = {emotion: int(np.sum(labels == emotion)) for emotion in label_encoder.classes_}
    n = len(labels)
    e_no = [counts[emotion] for emotion in label_encoder.classes_]

    return (
        n,
        counts.get('anger', 0),
        counts.get('love', 0),
        counts.get('fear', 0),
        counts.get('joy', 0),
        counts.get('sadness', 0),
        counts.get('surprise', 0),
        e_no,
        comments,
        labels.tolist()
    )
