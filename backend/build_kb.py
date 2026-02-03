import os
import json
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import re

# ------------------- Paths -------------------
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
DATA_JSON = KB_DIR / "data.json"
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"

# ------------------- Config -------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 200   # words per chunk (better for QA)
CHUNK_OVERLAP = 30

os.makedirs(KB_DIR, exist_ok=True)

# ------------------- Load JSON -------------------
if not DATA_JSON.exists():
    raise FileNotFoundError(f"{DATA_JSON} not found. Run convert_docx_to_json.py first!")

with open(DATA_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ------------------- Clean text -------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------- Chunking -------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# ------------------- Prepare texts -------------------
texts = []

for d in docs:
    if isinstance(d, str):
        d = clean_text(d)
        if len(d.split()) > 5:
            texts.extend(chunk_text(d))
    elif isinstance(d, dict) and d.get("text"):
        d = clean_text(d["text"])
        if len(d.split()) > 5:
            texts.extend(chunk_text(d))

if not texts:
    raise ValueError("No valid texts found in JSON!")

print(f"📄 Prepared {len(texts)} text chunks.")

# ------------------- Load embedding model -------------------
print("📦 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ------------------- Create embeddings -------------------
print("📊 Encoding chunks...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# ------------------- Build FAISS index -------------------
print("🔍 Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} vectors of dimension {dim}.")

# ------------------- Save index and texts -------------------
faiss.write_index(index, str(INDEX_PATH))
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(texts, f)

print("✅ Knowledge base successfully built and saved!")
