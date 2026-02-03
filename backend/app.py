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
from dotenv import load_dotenv
from openai import OpenAI

# ------------------ LOAD ENV ------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_KEY)

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

print("🔍 Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

# ------------------ FUNCTIONS ------------------
def embed_text(text):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

# ------------------ ROUTES ------------------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = (data.get("query") or data.get("message") or "").strip()
    top_k = int(data.get("top_k", 8))

    if not query:
        return jsonify({"error": "empty query"}), 400

    query_lower = query.lower()

    # ------------------ GREETING ------------------
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if query_lower in greetings:
        return jsonify({
            "query": query,
            "answer": "Hello! This is Ritz Media Bot answering.",
            "sources": []
        })

    # ------------------ EMBEDDING ------------------
    q_emb = embed_text(query_lower)

    # ------------------ FAISS SEARCH ------------------
    D, I = index.search(q_emb, top_k)
    top_texts = [chunks[i] for i in I[0] if i >= 0]

    if not top_texts:
        return jsonify({"query": query, "answer": "Sorry, no answer found.", "sources": []})

    # ------------------ PATTERN PRIORITY ------------------
    for text in top_texts:
        if any(word in query_lower for word in ["contact", "phone", "mobile"]):
            match = re.search(r"(\+?\d[\d\s-]{8,}\d)", text)
            if match:
                return jsonify({"query": query, "answer": match.group(1), "sources": top_texts})
        if any(word in query_lower for word in ["email", "mail"]):
            match = re.search(r"[\w\.-]+@[\w\.-]+", text)
            if match:
                return jsonify({"query": query, "answer": match.group(0), "sources": top_texts})
        if any(word in query_lower for word in ["address", "location"]):
            if "address" in text.lower():
                return jsonify({"query": query, "answer": text, "sources": top_texts})

    # ------------------ OPENAI CALL ------------------
    context = "\n".join(top_texts[:5])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say: "Sorry, I don't know."

Context:
{context}

Question: {query}
Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Ritz Media World. Use only the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    return jsonify({"query": query, "answer": answer, "sources": top_texts})

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
