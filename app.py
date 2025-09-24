# app.py  — Minimal Streamlit UI for Toxic Comment Detection (Keras)
# Expects: ./artifacts/text_cnn_toxic.keras, ./artifacts/tokenizer.json, ./artifacts/thresholds.npy (optional)

import os, re
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Toxic Comment Detection — Text CNN", page_icon="🧪", layout="centered")

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
MAX_LEN = 200
ART_DIR = Path("./artifacts")

# ---------- Cleaner (same as notebook) ----------
URL_RE  = re.compile(r'http\\S+|www\\.\\S+')
USER_RE = re.compile(r'@\\w+')
HTML_RE = re.compile(r'<.*?>')
SPACE_RE= re.compile(r'\\s+')
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = URL_RE.sub(' URL ', s)
    s = USER_RE.sub(' USER ', s)
    s = HTML_RE.sub(' ', s)
    s = SPACE_RE.sub(' ', s).strip()
    return s

# ---------- Cached loader (lazy TensorFlow import). No banners/UI here ----------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(art_dir: Path):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU inference

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model_path = art_dir / "text_cnn_toxic.keras"
    tok_path   = art_dir / "tokenizer.json"
    thr_path   = art_dir / "thresholds.npy"

    if not model_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts in {art_dir.resolve()} (need text_cnn_toxic.keras and tokenizer.json)."
        )

    # tokenizer_from_json expects a JSON string (not a dict)
    tok_text = tok_path.read_text()
    tokenizer = tokenizer_from_json(tok_text)

    thresholds = np.load(thr_path) if thr_path.exists() else np.full((len(LABELS),), 0.5, dtype=float)
    if thresholds.shape[0] != len(LABELS):
        thresholds = np.full((len(LABELS),), 0.5, dtype=float)  # safe fallback

    model = tf.keras.models.load_model(model_path, compile=False)
    return model, tokenizer, thresholds, pad_sequences

def predict_one(text: str, model, tokenizer, pad_sequences, thresholds: np.ndarray):
    cleaned = clean_text(text)
    seqs = tokenizer.texts_to_sequences([cleaned])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(X, verbose=0)[0]               # shape: (6,)
    preds = (probs >= thresholds).astype(int)            # 0/1 per label
    is_toxic = bool(preds.any())                         # any label active
    tox_score = float(probs.max())                       # simple overall score
    return probs, preds, is_toxic, tox_score

# ---------- Minimal UI ----------
st.title("🧪 Toxic Comment Detection — Text CNN")

# Load once; only show an error if something is wrong
try:
    model, tokenizer, thresholds, pad_sequences = load_model_and_tokenizer(ART_DIR)
except Exception as e:
    st.error("App initialization error. Check your artifacts folder.")
    st.exception(e)
    st.stop()

user_text = st.text_area("Enter a comment:", height=140, placeholder="Type or paste a comment…")
if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.info("Please enter some text.")
    else:
        probs, preds, is_toxic, tox_score = predict_one(user_text, model, tokenizer, pad_sequences, thresholds)
        st.markdown("### Result")
        if is_toxic:
            st.error(f"🚫 **TOXIC**  · score={tox_score:.3f}")
        else:
            st.success(f"✅ **NOT TOXIC**  · score={tox_score:.3f}")

        st.markdown("#### Per-label probabilities")
        st.dataframe(
            pd.DataFrame({
                "label": LABELS,
                "probability": np.round(probs, 4),
                "prediction": preds.astype(int),
            }),
            use_container_width=True
        )
