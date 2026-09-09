import streamlit as st
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# ── Config ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 10000
MAX_LEN    = 200

# ── Load word index once ─────────────────────────────────────────────────────
@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    word_to_idx = {w: i + 3 for w, i in word_index.items()}
    word_to_idx['<PAD>']   = 0
    word_to_idx['<START>'] = 1
    word_to_idx['<UNK>']   = 2
    return word_to_idx

import os

@st.cache_resource
def load_lstm():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'lstm_saved_model')
    return tf.saved_model.load(model_path)

# ── Preprocessing ────────────────────────────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode(text, word_to_idx):
    tokens = clean(text).split()
    encoded = [word_to_idx.get(w, 2) for w in tokens]   # 2 = <UNK>
    encoded = [i for i in encoded if i < VOCAB_SIZE]
    return pad_sequences([encoded], maxlen=MAX_LEN, padding='post', truncating='post')

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.caption("LSTM model trained on IMDb dataset")

review = st.text_area("Enter a movie review:", height=150,
                       placeholder="e.g. This movie was absolutely brilliant...")

if st.button("Predict", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        word_to_idx = load_word_index()
        model       = load_lstm()
        encoded     = encode(review, word_to_idx)
        input_tensor = tf.constant(encoded, dtype=tf.float32)
        infer = model.signatures['serving_default']
        output = infer(input_tensor)
        score = float(list(output.values())[0][0][0])
        label       = "Positive 😊" if score >= 0.5 else "Negative 😞"
        confidence  = score if score >= 0.5 else 1 - score

        col1, col2 = st.columns(2)
        col1.metric("Sentiment", label)
        col2.metric("Confidence", f"{confidence:.1%}")

        st.progress(float(score))
        st.caption(f"Raw score: {score:.4f}  (>0.5 = Positive)")
