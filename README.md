# 📝 RMW Docx Chatbot  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)](https://flask.palletsprojects.com/)  
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://faiss.ai/)  
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-SBERT-orange)](https://www.sbert.net/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

---

A chatbot that reads data from **DOCX files**, converts them into a structured **JSON knowledge base**, and answers user queries with relevant information.  
Built with **Python (Flask)**, **FAISS** for semantic search, and a simple **frontend (HTML, CSS, JS)**.  

---

## 🚀 Features
- 📄 Convert DOCX data into JSON format for structured knowledge.
- 🔍 Semantic search using **FAISS** for accurate answers.
- 💬 Chatbot interface (frontend + backend).
- ⚡ Lightweight Flask backend.
- 🌐 Easy deployment to Render, Hugging Face Spaces, or other platforms.

---

## 📂 Project Structure
rmw_docx_chatbot/
│── backend/
│ ├── app.py # Flask backend
│ ├── build_kb.py # Builds FAISS knowledge base
│ ├── convert_docx_to_json.py # Converts DOCX → JSON
│ ├── kb/
│ │ ├── data.json # Knowledge base data
│ │ ├── faiss_index.bin # FAISS index
│ │ ├── index.pkl # Pickled index
│ └── RMW Training Data 3.docx # Sample DOCX data
│
│── static/
│ ├── index.html # Frontend UI
│ ├── main.js # Chatbot logic
│ └── style.css # Styling
│
│── rmw-chatbot/ # (Optional extra folder with kb)
│── requirements.txt # Python dependencies
│── render.yaml # Deployment config for Render
│── .env.example # Example environment variables
│── README.md # Project documentation


---

## ⚙️ Installation

```bash
#
1. Clone the repo
git clone https://github.com/25061999/rmw_docx_chatbot.git
cd rmw_docx_chatbot

#
2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

▶️ Usage
# Build the knowledge base
python backend/build_kb.py

# Run the chatbot
python backend/app.py


Now open 👉 http://127.0.0.1:5000/ in your browser.

🌐 Deployment

Render → Supports deployment via render.yaml.

Hugging Face Spaces → Works with Gradio or Flask.

📌 Environment Variables

Copy .env.example → .env and add values:

OPENAI_API_KEY=your_api_key_here

🛠️ Tech Stack

Python 3.10+

Flask

FAISS

SentenceTransformers

HTML / CSS / JS

🤝 Contributing

Pull requests are welcome. Open an issue first to discuss changes.

📜 License

This project is licensed under the MIT License
.


---

👉 Steps for you:  
1. Replace your current **README.md** with the above content.  
2. Save it.  
3. Run:
```bash
git add README.md
git commit -m "Update README with badges"
git push origin main
