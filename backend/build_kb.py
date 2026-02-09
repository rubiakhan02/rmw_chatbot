import os
import pickle
import faiss
import re
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
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# -------- READ WEBSITE --------
def read_website_txt(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------- SMART CHUNKING --------
def smart_chunk(text, size=500, overlap=80):

    # split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + s

    if current:
        chunks.append(current.strip())

    return chunks


# -------- MAIN --------
print("Loading data...")

docx_file = BASE_DIR / "RMW Training Data 3.docx"
website_file = BASE_DIR / "website_data.txt"

docx_text = read_docx(docx_file)
website_text = read_website_txt(website_file)

# Tag sources
docx_text = "[DOCX]\n" + docx_text
website_text = "[WEBSITE]\n" + website_text

combined_text = docx_text + "\n" + website_text

print("Chunking text...")
chunks = smart_chunk(combined_text)

print("Total chunks:", len(chunks))

print("Creating embeddings...")
embeddings = model.encode(chunks, convert_to_numpy=True)

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("Saving index...")
faiss.write_index(index, str(INDEX_PATH))

with open(PICKLE_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("✅ DONE — Improved KB built")
