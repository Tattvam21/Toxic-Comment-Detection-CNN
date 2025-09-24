# app.py
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# Keep TF import inside cached loader (faster cold start on Streamlit)
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(art_dir: Path, max_len: int):
    # Lazy TF import
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU inference
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    # Load model
    model_path = art_dir / "text_cnn_toxic.keras"
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load tokenizer
    tok_json = json.loads((art_dir / "tokenizer.json").read_text())
    tokenizer = tokenizer_from_json(tok_json)

    # Load thresholds (or default 0.5)
    thr_path = art_dir / "thresholds.npy"
    if thr_path.exists():
        thresholds = np.load(thr_path)
    else:
        thresholds = np.full((6,), 0.5, dtype=float)

    return model, tokenizer, thresholds, pad_sequences, tf

# ---- Config ----
st.set_page_config(page_title="Toxic Comment Detection (Text CNN)", page_icon="🧪", layout="wide")
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
MAX_LEN = 200
ART_DIR = Path("./artifacts")

# ---- Load resources ----
with st.spinner("Loading model & tokenizer..."):
    model, tokenizer, thresholds, pad_sequences, tf = load_model_and_tokenizer(ART_DIR, MAX_LEN)

# ---- Cleaner (same as notebook) ----
import re
URL_RE = re.compile(r'http\S+|www\.\S+')
USER_RE = re.compile(r'@\w+')
HTML_RE = re.compile(r'<.*?>')
SPACE_RE= re.compile(r'\s+')
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = URL_RE.sub(' URL ', s)
    s = USER_RE.sub(' USER ', s)
    s = HTML_RE.sub(' ', s)
    s = SPACE_RE.sub(' ', s).strip()
    return s

def predict_texts(texts, thresholds=None):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(X, verbose=0)
    thr = thresholds if thresholds is not None else np.full((len(LABELS),), 0.5)
    preds = (probs >= thr).astype(int)
    return probs, preds

# ---- UI ----
st.title("🧪 Toxic Comment Detection — Text CNN")
st.caption("Keras Text-CNN with per-class thresholds tuned on validation. Not a production content-moderation system.")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Single text")
    user_text = st.text_area("Enter a comment:", height=140, placeholder="Type or paste a comment here…")
    if st.button("Analyze"):
        if user_text.strip():
            probs, preds = predict_texts(user_text, thresholds=thresholds)
            prob_row = probs[0]
            pred_row = preds[0]
            df_out = pd.DataFrame({"label": LABELS,
                                   "probability": np.round(prob_row, 4),
                                   "prediction": pred_row.astype(int)})
            st.dataframe(df_out, use_container_width=True)
        else:
            st.info("Please enter some text.")

with col2:
    st.subheader("Batch (CSV)")
    uploaded = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])
    if uploaded is not None:
        try:
            test_df = pd.read_csv(uploaded)
            assert 'comment_text' in test_df.columns, "CSV must contain a 'comment_text' column."
            st.write("Rows:", len(test_df))
            if st.button("Run batch inference"):
                probs, preds = predict_texts(test_df['comment_text'].tolist(), thresholds=thresholds)
                probs_df = pd.DataFrame(probs, columns=[f"p_{l}" for l in LABELS])
                preds_df = pd.DataFrame(preds, columns=[f"y_{l}" for l in LABELS])
                out_df = pd.concat([test_df.reset_index(drop=True), probs_df, preds_df], axis=1)
                st.success("Done.")
                st.dataframe(out_df.head(50), use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="toxic_predictions.csv")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with st.expander("Thresholds (tuned on validation)"):
    tdf = pd.DataFrame({"label": LABELS, "threshold": np.round(thresholds, 3)})
    st.dataframe(tdf, use_container_width=True)
    st.caption("You tuned these in the notebook. Change them there and re-export thresholds.npy to update.")

st.markdown("---")
st.caption("⚠️ For research/education only. For real moderation, use audited pipelines and policies.")
