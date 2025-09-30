import json
import docx
from pathlib import Path
import re

DOCX_PATH = Path("backend/RMW Training Data 3.docx")
JSON_PATH = Path("backend/kb/data.json")

def split_into_chunks(text, max_words=80):
    """Split text into chunks of max_words each."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def clean_text(text):
    """Remove extra spaces and normalize line breaks."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def convert_docx_to_json():
    print("📄 Reading DOCX...")
    doc = docx.Document(DOCX_PATH)
    
    paragraphs = []
    for p in doc.paragraphs:
        text = clean_text(p.text)
        if not text:
            continue
        # Split long paragraphs
        chunks = split_into_chunks(text)
        paragraphs.extend(chunks)
    
    data = {"chunks": paragraphs}
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(paragraphs)} chunks to {JSON_PATH}")

if __name__ == "__main__":
    convert_docx_to_json()
