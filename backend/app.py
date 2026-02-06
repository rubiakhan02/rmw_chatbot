import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pickle
import faiss
import re
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
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
model = SentenceTransformer(MODEL_NAME)

with open(PICKLE_PATH, "rb") as f:
    chunks = pickle.load(f)

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

    if not query:
        return jsonify({"error": "empty query"}), 400

    query_lower = query.lower()

    # ===== NORMAL CONVERSATION HANDLING =====
    ack_words = ["ok", "okay", "thanks", "thank you", "good", "fine", "great"]
    if query_lower in ack_words:
        return jsonify({
            "query": query,
            "answer": "You're welcome 😊 How can I help you with Ritz Media World?",
            "sources": []
        })

    small_talk = ["how are you", "who are you", "what are you", "your age", "what is your age"]
    if any(x in query_lower for x in small_talk):
        return jsonify({
            "query": query,
            "answer": "I’m an AI assistant for Ritz Media World, here to help you with company information and services.",
            "sources": []
        })

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if query_lower in greetings:
        return jsonify({
            "query": query,
            "answer": "Hello! This is Ritz Media Bot answering.",
            "sources": []
        })

    # ===== PRICE QUESTIONS =====
    if any(w in query_lower for w in ["price", "cost", "charge", "fees", "rate"]):
        return jsonify({
            "query": query,
            "answer": "Pricing depends on project requirements. Please contact Ritz Media World directly for an exact quotation.",
            "sources": []
        })

    # ===== VECTOR SEARCH =====
    q_emb = embed_text(query_lower)
    D, I = index.search(q_emb, 12)
    top_texts = [chunks[i] for i in I[0] if i >= 0]

    if not top_texts:
        return jsonify({
            "query": query,
            "answer": "I don’t have verified information on that right now. Please contact Ritz Media World directly for accurate details.",
            "sources": []
        })

    # ===== PHONE =====
    if any(w in query_lower for w in ["phone", "call", "contact"]):
        for text in top_texts:
            match = re.search(r"(\+?\d[\d\s-]{8,}\d)", text)
            if match:
                return jsonify({"query": query, "answer": match.group(1), "sources": []})

    # ===== EMAIL =====
    if "email" in query_lower:
        for text in top_texts:
            match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
            if match:
                return jsonify({"query": query, "answer": match.group(0), "sources": []})

    # ===== ADDRESS =====
    if any(w in query_lower for w in ["address", "location", "located", "where"]):
        for text in top_texts:
            if "noida" in text.lower():
                address_match = re.search(r"(402–404.*?India)", text)
                if address_match:
                    return jsonify({"query": query, "answer": address_match.group(1), "sources": []})

    # ===== SERVICES LIST =====
    if "list" in query_lower and "service" in query_lower:
        services = [
            "Digital Marketing",
            "Creative & Branding",
            "Print Advertising",
            "Radio Advertising",
            "Content Marketing",
            "Web Development",
            "Influencer Marketing",
            "Celebrity Endorsements",
            "Media Planning and Buying",
            "Public Relations and Campaign Management"
        ]
        return jsonify({
            "query": query,
            "answer": "Here are the main services provided by Ritz Media World:\n- " + "\n- ".join(services),
            "sources": []
        })

    # ===== OPENAI FOR DESCRIPTION =====
    context = "\n".join(top_texts[:4])

    prompt = f"""
Answer based on the context below.
Keep the answer short and clear (1-2 sentences).

Context:
{context}

Question: {query}
Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Ritz Media World."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=120
    )

    answer = response.choices[0].message.content.strip()
    answer = answer.split("\n")[0]
    if "." in answer:
        answer = answer.split(".")[0] + "."

    return jsonify({"query": query, "answer": answer, "sources": []})

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
