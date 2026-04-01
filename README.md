# 📄 SupplyChain RAG Assistant (Local Retrieval-Augmented Generation)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline over PDF documents using local models.

It allows you to:
- Index multiple PDF documents
- Retrieve relevant content using vector similarity
- Generate grounded answers using an LLM

---

## 🚀 Features

- Load and process multiple PDFs
- Intelligent text chunking with overlap
- Embedding generation using local Ollama models
- Vector similarity search using FAISS
- Question answering grounded in document context
- Source-aware responses

---

## 🏗️ Architecture

```
PDFs
  ↓
Text Extraction (pypdf)
  ↓
Chunking
  ↓
Embeddings (Ollama)
  ↓
FAISS Vector Index

User Query
  ↓
Query Embedding
  ↓
Similarity Search (FAISS)
  ↓
Top-K Relevant Chunks
  ↓
LLM (Ollama)
  ↓
Final Answer
```

---

## 📁 Project Structure

```
rag-pdf-demo/
│
├── data/
│   └── pdfs/              # Input PDF files
│
├── storage/
│   ├── faiss.index        # Vector index
│   └── metadata.json      # Chunk metadata
│
├── ingest.py              # Builds vector database
├── query.py               # Query interface
├── rag_utils.py           # Core RAG logic
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-pdf-demo
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Make sure Ollama is running locally:

```bash
ollama serve
```

### 5. Pull required models

```bash
ollama pull embeddinggemma
ollama pull llama3.2
```

---

## 📥 Add Documents

Place your PDF files inside:

```
data/pdfs/
```

---

## 🧠 Build the Vector Index

```bash
python ingest.py
```

This will:
- Extract text from PDFs
- Split into chunks
- Generate embeddings
- Store them in FAISS

---

## 💬 Query the System

```bash
python query.py
```

**Example questions:**
- What is cold chain logistics?
- Summarize the main idea of the document
- What are the key challenges mentioned?

---

## 🧪 Example Output

```
Top retrieved chunks:
1. cold_chain_logistics_framework.pdf | chunk 1

Answer:
A cold chain is a temperature-controlled supply chain used to store and transport perishable products.

Sources:
- ai_ml_in_coldchain.pdf (chunk 17)
```

---

## 🔍 Key Concepts

**Embeddings**
Text is converted into numerical vectors representing semantic meaning.

**Vector Database (FAISS)**
Stores embeddings and performs fast similarity search.

**RAG (Retrieval-Augmented Generation)**
Instead of training the model, relevant document chunks are retrieved and used as context for answer generation.

---

## ⚠️ Notes

- This implementation uses `pypdf` for text extraction. Some PDFs may produce noisy text.
- For production systems, consider:
  - `PyMuPDF` (fitz)
  - OCR for scanned documents
- Embeddings are generated locally using Ollama (no external APIs required).
