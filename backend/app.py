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
from flask_cors import CORS
import numpy as np

# ------------------ PATHS ------------------
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ------------------ FLASK APP ------------------
app = Flask(__name__, static_folder="../static", static_url_path="/static")
CORS(app)

# ------------------ LOAD MODEL & KB ------------------
print("📦 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("📄 Loading chunks...")
with open(PICKLE_PATH, "rb") as f:
    chunks = pickle.load(f)
chunks_lower = [c.lower() for c in chunks]

print("🔍 Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

# ------------------ FUNCTIONS ------------------
def embed_text(text):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

def rerank(query_emb, top_texts):
    """Rerank top texts by cosine similarity to query."""
    top_emb = model.encode(top_texts, convert_to_numpy=True)
    faiss.normalize_L2(top_emb)
    sims = np.dot(top_emb, query_emb.T).flatten()
    sorted_idx = np.argsort(-sims)
    return [top_texts[i] for i in sorted_idx], sims[sorted_idx]

# ------------------ ROUTES ------------------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 10))  # default top 10

    if not query:
        return jsonify({"error": "empty query"}), 400

    query_lower = query.lower()

    # ------------------ GREETING ------------------
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(word in query_lower for word in greetings):
        return jsonify({
            "query": query,
            "answer": "Hello! This is Ritz Media Bot answering.",
            "sources": []
        })

    # ------------------ EMBEDDING ------------------
    q_emb = embed_text(query_lower)

    # ------------------ FAISS SEARCH ------------------
    D, I = index.search(q_emb, top_k)
    top_results = [
        {"score": float(D[0][i]), "text": chunks[I[0][i]]} 
        for i in range(len(I[0])) if I[0][i] >= 0
    ]

    if not top_results:
        return jsonify({"query": query, "answer": "Sorry, no answer found.", "sources": []})

    # ------------------ PATTERN PRIORITY ------------------
    for r in top_results:
        text = r["text"]
        if any(word in query_lower for word in ["contact", "phone", "mobile"]):
            match = re.search(r"(\+?\d[\d\s-]{8,}\d)", text)
            if match:
                return jsonify({"query": query, "answer": match.group(1), "sources": top_results})
        if any(word in query_lower for word in ["email", "mail"]):
            match = re.search(r"[\w\.-]+@[\w\.-]+", text)
            if match:
                return jsonify({"query": query, "answer": match.group(0), "sources": top_results})
        if any(word in query_lower for word in ["address", "location"]):
            if "address" in text.lower():
                return jsonify({"query": query, "answer": text, "sources": top_results})

    # ------------------ RERANK ------------------
    top_texts = [r["text"] for r in top_results]
    reranked_texts, sims = rerank(q_emb, top_texts)

    combined_text = reranked_texts[0] if reranked_texts else top_texts[0]
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    answer = " ".join(sentences[:2]) if sentences else "No answer found."

    return jsonify({"query": query, "answer": answer, "sources": top_results})

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
