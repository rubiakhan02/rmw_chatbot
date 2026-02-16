import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pickle
import faiss
import re
import os
import requests
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


# ⭐ SELECT BEST CONTEXT (NEW)
def select_best_chunks(indices):
    texts = [chunks[i] for i in indices if i >= 0]

    # Rank by richness (longer chunks first)
    texts.sort(key=len, reverse=True)

    # Return top meaningful chunks
    return texts[:6]


# ------------------ ROUTES ------------------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


# ================= LEAD PROXY =================
@app.route("/submit-lead", methods=["POST"])
def submit_lead():

    data = request.json

    formatted_message = (
        f"Source: Chatbot\n"
        f"Selected Service: {data.get('service')}\n\n"
        f"User Message:\n{data.get('message') or 'N/A'}"
    )

    payload = {
        "etype": "ContactUs",
        "name": data.get("name"),
        "phone": data.get("phone"),
        "email": data.get("email"),
        "message": formatted_message
    }

    try:
        r = requests.post(
            "https://ritzmediaworld.com/api/system-settings/contact-enquiry",
            json=payload,
            timeout=12
        )
        return jsonify(r.json())

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ================= CHAT =================
@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    query = (data.get("query") or data.get("message") or "").strip()

    if not query:
        return jsonify({"error": "empty query"}), 400

    query_lower = query.lower()

    # ===== SIMPLE RESPONSES =====
    if query_lower in ["ok","okay","thanks","thank you","good","fine","great"]:
        return jsonify({"query": query,
                        "answer": "Glad I could help!😊 Let me know if there's anything else I can assist you with.",
                        "sources": []})

    if any(x in query_lower for x in ["who are you","what are you"]):
        return jsonify({"query": query,
                        "answer": "I’m Ruby, AI assistant for Ritz Media World.",
                        "sources": []})

    if query_lower in ["hi","hello","hey"]:
        return jsonify({"query": query,
                        "answer": "Hello! This is Ruby answering.",
                        "sources": []})

    if any(w in query_lower for w in ["price","cost","charge","pricing","rates","rate","charges","how much"]):
        return jsonify({
            "query": query,
            "answer": "Pricing depends on requirements. Please contact the team for a quote.",
            "sources": []
        })

    
    try:
        # ===== VECTOR SEARCH (IMPROVED) =====
        q_emb = embed_text(query_lower)
        D, I = index.search(q_emb, 20)

        # SAFE similarity score check (prevents crash)
        top_score = 0
        if len(D) > 0 and len(D[0]) > 0:
            top_score = float(D[0][0])


        # ⭐ threshold tuning
        if top_score < 0.35:
            best_chunks = []
        else:
            best_chunks = select_best_chunks(I[0])


        if not best_chunks:

            # ===== GENERAL OPENAI FALLBACK (Conversation) =====
            try:
                system_instruction = {
                    "role": "system",
                    "content": "You are Ruby, a friendly AI assistant for Ritz Media World. Start a polite conversation if the user greets you. IMPORTANT: If the user asks for 'your' contact (email/phone/address), provide Ritz Media World's contact details. Never say you don't have personal info."
                }
                
                # Get history from request
                history = data.get("history", [])
                
                # Construct messages: System + History
                # If history is empty/invalid, fall back to just query
                if isinstance(history, list) and len(history) > 0:
                    messages = [system_instruction] + history
                else:
                    messages = [
                        system_instruction,
                        {"role": "user", "content": query}
                    ]

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI Fallback Error: {e}")
                answer = "I'm having trouble thinking right now. Please try again later."


            return jsonify({
                "query": query,
                "answer": answer,
                "sources": []
            })



        # ===== CONTACT EXTRACTION =====
        for text in best_chunks:
            if "phone" in query_lower:
                match = re.search(r"(\+?\d[\d\s-]{8,}\d)", text)
                if match:
                    return jsonify({"query": query, "answer": match.group(1), "sources": []})

            if "email" in query_lower:
                match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
                if match:
                    return jsonify({"query": query, "answer": match.group(0), "sources": []})

        # ===== OPENAI ANSWER (Context-Aware) =====
        context = "\n".join(best_chunks)

        prompt = f"""
    You are Ruby, AI assistant for Ritz Media World.
    Be helpful, conversational, and professional.
    Explain clearly in a polite tone.


    RULES:
    - Primary Source: Use the Context below.
    - If the user is just chatting (e.g. "nice weather"), reply politely without context.
    - IMPORTANT: If the user asks for 'your' contact details (email, phone, address), ALWAYS provide Ritz Media World's contact details from the context.
    - When asked about a specific sub-service (like SEO, Google Ads, Social Media Marketing, etc.), FIRST provide detailed information about that specific sub-service from the context.
    - ONLY AFTER providing the specific sub-service details, mention that "We offer this service under [category name]" if relevant.
    - DO NOT list all services in a category when asked about one specific sub-service.
    - Be concise and focused on what the user specifically asked about.

    Context:
    {context}

    Question: {query}
    Answer:
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You help users with Ritz Media World info."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({"query": query, "answer": answer, "sources": []})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e), "answer": "Something went wrong internally."}), 500


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
