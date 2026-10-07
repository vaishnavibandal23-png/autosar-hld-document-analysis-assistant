# AUTOSAR HLD Document Analysis Assistant

A beginner-friendly RAG application for extracting, indexing, searching and answering questions about AUTOSAR High-Level Design (HLD) PDF documents.

## Features

- PDF ingestion with PyMuPDF
- Page-aware and section-aware chunking
- Sentence Transformer embeddings
- FAISS vector search
- Grounded answers with source markers
- Page and section citations
- Basic citation/retrieval validation
- Optional local LLM through Ollama
- Safe extractive fallback when Ollama is unavailable
- Streamlit web interface
- Local/private-first architecture

## Project structure

```text
autosar-hld-assistant/
├── app.py
├── config.py
├── document_processor.py
├── vector_store.py
├── llm.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── scripts/
│   └── rebuild_index.py
├── data/
│   ├── documents/
│   └── index/
└── outputs/
```

## Requirements

- Windows 10/11, macOS or Linux
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended
- Internet connection for the first installation/model download
- Optional: Ollama for local LLM generation

## 1. Create virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add HLD documents

Put approved PDF documents into:

```text
data/documents/
```

Do not commit confidential company HLD files to a public GitHub repository.

## 4. Build the knowledge base

```powershell
python scripts/rebuild_index.py
```

The first run downloads the embedding model. The generated FAISS index and metadata remain local.

## 5. Run the application

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## 6. Optional local LLM with Ollama

Install Ollama separately and download a model:

```powershell
ollama pull llama3.2:3b
```

Then make sure Ollama is running:

```powershell
ollama serve
```

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If Ollama is unavailable, the application still runs and returns retrieved evidence using the safe extractive fallback.

## Example questions

- Which software components are defined in the HLD?
- What interfaces are associated with a specific component?
- Which ports are provided and required?
- What dependency exists between two components?
- What functional flow is described for a specific signal?

## Evaluation

For an academic evaluation, prepare an expert-approved ground-truth set of questions and record:

- Answer relevance / faithfulness
- Retrieval hit rate
- Citation accuracy
- Average response latency
- False positives and missed dependencies
- Failure cases

Do not report placeholder or sample metrics as measured results.

## Privacy

Keep confidential HLD PDFs, `.env`, local indexes and generated outputs outside public Git repositories. The application is designed for local/private deployment.

## Limitations

- PDF text extraction is strongest for digitally generated PDFs.
- Complex architecture diagrams are not automatically understood.
- OCR for scanned PDFs requires an external OCR engine such as Tesseract.
- The optional LLM must still be reviewed by an engineer for architecture decisions.
