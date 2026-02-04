import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from docx import Document

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
KB_DIR.mkdir(exist_ok=True)

INDEX_PATH = KB_DIR / "faiss_index.bin"
PICKLE_PATH = KB_DIR / "index.pkl"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# -------- READ DOCX --------
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# -------- READ WEBSITE DATA --------
def read_website_txt(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -------- CHUNK TEXT --------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# -------- MAIN --------
print("Loading data...")

docx_file = BASE_DIR / "RMW Training Data 3.docx"
website_file = BASE_DIR / "website_data.txt"

docx_text = read_docx(docx_file)
website_text = read_website_txt(website_file)

combined_text = docx_text + "\n" + website_text

print("Chunking text...")
chunks = chunk_text(combined_text)

print("Total chunks:", len(chunks))

print("Creating embeddings...")
embeddings = model.encode(chunks, convert_to_numpy=True)

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("Saving index and chunks...")
faiss.write_index(index, str(INDEX_PATH))

with open(PICKLE_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("DONE. Knowledge base updated with DOCX + Website data.")
