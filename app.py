# app.py
# Streamlit Toxic Comment Detection — Text CNN (Keras)
# Expects artifacts in ./artifacts : text_cnn_toxic.keras, tokenizer.json, thresholds.npy (optional)

import os, sys, re
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page & constants ----------------
st.set_page_config(page_title="Toxic Comment Detection — Text CNN", page_icon="🧪", layout="wide")

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
MAX_LEN = 200
ART_DIR = Path("./artifacts")

# ---------------- Cleaner (same as notebook) ----------------
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

# ---------------- Cached loader (lazy TensorFlow import) ----------------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(art_dir: Path):
    # Quieter TF, CPU-only (good defaults for Streamlit & local)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model_path = art_dir / "text_cnn_toxic.keras"
    tok_path   = art_dir / "tokenizer.json"
    thr_path   = art_dir / "thresholds.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing artifact: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing artifact: {tok_path}")

    # ✅ IMPORTANT: tokenizer_from_json expects a JSON **string**
    tok_text = tok_path.read_text()
    tokenizer = tokenizer_from_json(tok_text)

    # thresholds are optional; default to 0.5 if absent
    thresholds = np.load(thr_path) if thr_path.exists() else np.full((len(LABELS),), 0.5, dtype=float)
    if thresholds.shape[0] != len(LABELS):
        raise ValueError(f"thresholds.npy shape {thresholds.shape} does not match {len(LABELS)} labels")

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    return model, tokenizer, thresholds, pad_sequences

def predict_one(text: str, model, tokenizer, pad_sequences, thresholds: np.ndarray):
    cleaned = clean_text(text)
    seqs = tokenizer.texts_to_sequences([cleaned])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(X, verbose=0)[0]                 # shape: (6,)
    preds = (probs >= thresholds).astype(int)              # 0/1 per label
    is_toxic = bool(preds.any())                           # overall decision: any label positive
    tox_score = float(probs.max())                         # simple overall score = max label prob
    return probs, preds, is_toxic, tox_score

# ---------------- Header ----------------
st.title("🧪 Toxic Comment Detection — Text CNN")
st.caption("Type a comment and get a TOXIC / NOT TOXIC decision (and per-label probabilities).")

# ---------------- Environment & artifacts (helpful for debugging) ----------------
with st.expander("Environment & Artifacts"):
    st.write("**Python:**", sys.version)
    st.write("**Working directory:**", Path.cwd())
    st.write("**Artifacts dir:**", ART_DIR.resolve())
    try:
        st.write("**Files in artifacts:**", [p.name for p in ART_DIR.iterdir()])
    except Exception as e:
        st.warning(f"Could not list artifacts: {e}")

# ---------------- Load resources with visible errors ----------------
with st.spinner("Loading model & tokenizer..."):
    try:
        model, tokenizer, thresholds, pad_sequences = load_model_and_tokenizer(ART_DIR)
        st.success("Model & tokenizer loaded.")
        load_error = None
    except Exception as e:
        load_error = e
        st.error("Startup error while loading artifacts / TensorFlow.")
        st.exception(e)

# ---------------- Simple center UI ----------------
user_text = st.text_area("Enter a comment:", height=140, placeholder="e.g., You're clueless and incompetent, just stop talking.")

# Optional: quick global threshold override
thr_default = float(thresholds.mean()) if isinstance(thresholds, np.ndarray) else 0.5
thr_any = st.slider("Decision threshold (applied per label)", 0.1, 0.9, thr_default, 0.05)
if isinstance(thresholds, np.ndarray):
    thresholds = np.full_like(thresholds, thr_any, dtype=float)

if st.button("Predict", type="primary"):
    if load_error is not None:
        st.warning("Model not available — fix the error above, then try again.")
    elif not user_text.strip():
        st.info("Please enter some text.")
    else:
        probs, preds, is_toxic, tox_score = predict_one(user_text, model, tokenizer, pad_sequences, thresholds)
        if is_toxic:
            st.error(f"🚫 TOXIC  (score={tox_score:.3f})")
        else:
            st.success(f"✅ NOT TOXIC  (score={tox_score:.3f})")

        # Per-label breakdown
        out = pd.DataFrame({
            "label": LABELS,
            "probability": np.round(probs, 4),
            "prediction": preds.astype(int),
        })
        st.dataframe(out, use_container_width=True)
