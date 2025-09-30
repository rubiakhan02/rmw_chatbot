import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import pickle
import faiss
import re
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

app = Flask(__name__, static_folder="../static", static_url_path="/static")

# Load model
print("📦 Loading model...")
model = SentenceTransformer(MODEL_NAME)

# Load chunks and FAISS index
print("📄 Loading chunks...")
with open(PICKLE_PATH, "rb") as f:
    chunks = pickle.load(f)

print("🔍 Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

def embed_text(text):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "empty query"}), 400

    q_emb = embed_text(query)
    D, I = index.search(q_emb, top_k)
    results = [{"score": float(score), "text": chunks[idx]} 
               for score, idx in zip(D[0], I[0]) if idx >= 0]

    if not results:
        return jsonify({"query": query, "answer": "Sorry, no answer found.", "sources": []})

    # Pattern-priority (phone/email/address)
    query_lower = query.lower()
    for r in results:
        if "contact" in query_lower or "phone" in query_lower or "mobile" in query_lower:
            match = re.search(r"(\+?\d[\d\s-]{8,}\d)", r["text"])
            if match:
                return jsonify({"query": query, "answer": match.group(1), "sources": results})
        if "email" in query_lower or "mail" in query_lower:
            match = re.search(r"[\w\.-]+@[\w\.-]+", r["text"])
            if match:
                return jsonify({"query": query, "answer": match.group(0), "sources": results})
        if "address" in query_lower or "location" in query_lower:
            if "address" in r["text"].lower():
                return jsonify({"query": query, "answer": r["text"], "sources": results})

    # Default: return first 2 sentences from top chunks
    combined_text = " ".join([r["text"] for r in results])
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    answer = " ".join(sentences[:2]) if sentences else "No answer found."

    return jsonify({"query": query, "answer": answer, "sources": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

