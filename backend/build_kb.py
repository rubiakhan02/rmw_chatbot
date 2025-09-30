import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import gradio as gr
import json
import pickle
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"

# Load model
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

# Load chunks and FAISS index
with open(PICKLE_PATH, "rb") as f:
    chunks = pickle.load(f)
index = faiss.read_index(str(INDEX_PATH))

# Function to embed query
def embed_text(text):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

# Chat function
def chat(query, top_k=5):
    q_emb = embed_text(query)
    D, I = index.search(q_emb, top_k)
    results = [{"score": float(score), "text": chunks[idx]} 
               for score, idx in zip(D[0], I[0]) if idx >= 0]

    if not results:
        return "Sorry, no answer found."

    # Pattern-priority answers
    query_lower = query.lower()
    for r in results:
        if any(x in query_lower for x in ["contact", "phone", "mobile"]):
            match = re.search(r"(\+?\d[\d\s-]{8,}\d)", r["text"])
            if match:
                return match.group(1)
        if any(x in query_lower for x in ["email", "mail"]):
            match = re.search(r"[\w\.-]+@[\w\.-]+", r["text"])
            if match:
                return match.group(0)
        if any(x in query_lower for x in ["address", "location"]):
            if "address" in r["text"].lower():
                return r["text"]

    # Default: first 2 sentences
    combined_text = " ".join([r["text"] for r in results])
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    answer = " ".join(sentences[:2]) if sentences else "No answer found."
    return answer

# Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(lines=2, placeholder="Ask me something...")],
    outputs="text",
    title="RMW Chatbot",
    description="Ask anything from the knowledge base. Supports phone/email/address extraction."
)

iface.launch()
