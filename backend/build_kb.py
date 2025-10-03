import os
import json
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# ------------------- Paths -------------------
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
DATA_JSON = KB_DIR / "data.json"
INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"

# ------------------- Config -------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Strong embedding model
MAX_WORDS_PER_CHUNK = 40  # adjust chunk size for better granularity

os.makedirs(KB_DIR, exist_ok=True)

# ------------------- Load DOCX JSON -------------------
if not DATA_JSON.exists():
    raise FileNotFoundError(f"{DATA_JSON} not found. Run convert_docx_to_json.py first!")

with open(DATA_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ------------------- Prepare texts -------------------
texts = []
for d in docs:
    if isinstance(d, str):
        texts.append(d.strip())
    elif isinstance(d, dict) and d.get("text"):
        texts.append(d["text"].strip())

if not texts:
    raise ValueError("No valid texts found in JSON!")

print(f"📄 Loaded {len(texts)} chunks.")

# ------------------- Load embedding model -------------------
print("📦 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ------------------- Create embeddings -------------------
print("📊 Encoding chunks...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# ------------------- Build FAISS index -------------------
print("🔍 Building FAISS index...")
dim = embeddings.shape[1]  # Ensure correct dimension
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors of dimension {dim}.")

# ------------------- Save index and texts -------------------
faiss.write_index(index, str(INDEX_PATH))
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(texts, f)

print("✅ Knowledge base successfully built and saved!")
