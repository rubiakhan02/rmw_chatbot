import os
import json
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# ------------------- Paths -------------------
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
DATA_JSON = KB_DIR / "data.json"          # Your knowledge base JSON
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# Ensure kb directory exists
os.makedirs(KB_DIR, exist_ok=True)

# ------------------- Load JSON -------------------
if not DATA_JSON.exists():
    raise FileNotFoundError(f"{DATA_JSON} not found. Please create your JSON first!")

with open(DATA_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ------------------- Process documents -------------------
texts = []
for d in docs:
    if isinstance(d, dict) and d.get("text"):
        texts.append(d["text"].strip())
    elif isinstance(d, str):
        texts.append(d.strip())

if not texts:
    raise ValueError("No valid texts found in JSON!")

print(f"📄 Loaded {len(texts)} documents.")

# ------------------- Load model and embed -------------------
print("📦 Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

# ------------------- Build FAISS index -------------------
print("🔍 Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ------------------- Save index and texts -------------------
faiss.write_index(index, str(INDEX_PATH))  # Windows-safe
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(texts, f)

print("✅ Knowledge base built successfully!")
