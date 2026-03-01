
# Career Recommendation System (Retrieval-Augmented Generation)

## Overview
This project implements the **retrieval component** of a Retrieval-Augmented Generation (RAG) pipeline.
It focuses on **semantic document retrieval and evaluation**, enabling accurate search over large collections
of resumes (CVs) and career-related documents.

The system indexes documents using vector embeddings and compares multiple **chunking strategies**
to analyze their impact on retrieval quality.

> ⚠️ Note  
> This repository focuses on **retrieval and evaluation only**.  
> It does **not** include LLM-based response generation or chat interfaces.

---

## Key Features
- Semantic search over CVs and PDF documents
- Multiple chunking strategies:
  - Paragraph-based chunking
  - Semantic chunking
  - KamradtModifiedChunker
- Vector similarity search using **ChromaDB**
- Dense embeddings via **SentenceTransformers (MiniLM-L6-v2)**
- Quantitative evaluation using IR metrics:
  - Precision
  - Mean Average Precision (MAP)
  - nDCG

---

## Tech Stack
- **Python 3.9+**
- SentenceTransformers
- ChromaDB (Vector Database)
- Pandas, NumPy
- scikit-learn
- PyPDF

---

## Installation
Clone the repository and install dependencies:

```bash
pip install chromadb pandas pypdf sentence-transformers scikit-learn numpy
```

---

## ⚡ Quick Start (Recommended)

This project is a **pipeline**, not a single script.  
Follow the steps below **in order** to run and test the system correctly.

### Step 1: Prepare the Dataset
You may use either:
- A Kaggle resume dataset (CSV format), or
- Your own directory of PDF resumes

**Kaggle Dataset Requirements**
- File: `UpdatedResumeDataSet.csv`
- Required column: `Resume`

Place the dataset under:
```
data/resumes/
```

---

### Step 2: Build the Corpus
Convert resumes into a structured JSON corpus:

```python
build_kaggle_resume_corpus(
    input_csv_path="data/resumes/UpdatedResumeDataSet.csv",
    output_json_path="data/careers.json",
    text_column="Resume"
)
```

---

### Step 3: Index Documents in ChromaDB
Choose **one** chunking strategy.

**Semantic Chunking**
```python
build_pdfs_index_from_json(
    json_path="data/careers.json",
    collection_name="pdf_semantic"
)
```

**Paragraph-Based Chunking**
```python
build_pdfs_index_paragraph_from_json(
    json_path="data/careers.json",
    collection_name="pdf_paragraph"
)
```

---

### Step 4: Run Retrieval
Query the indexed documents:

```python
retrieve_from_pdfs(
    query_text="Python backend developer experience",
    k=5,
    collection_name="pdf_semantic"
)
```

---

## Evaluation
Retrieval quality is evaluated using standard Information Retrieval metrics:

- **Precision** – relevance of top-k retrieved documents
- **MAP (Mean Average Precision)** – ranking quality across queries
- **nDCG** – ranking usefulness considering position

Chunking strategies are compared using identical query sets to ensure fairness.

---

## Project Structure
```
.
├── data/
│   ├── resumes/
│   └── careers.json
├── indexing/
├── retrieval/
├── evaluation/
├── utils/
└── README.md
```

---

## Use Cases
- Career recommendation systems
- Resume semantic search
- HR document retrieval
- Academic research on chunking strategies in RAG pipelines

---

## Author
**Sama Shalabi**  
Junior AI Engineer  

---

## License
This project is provided for academic and educational purposes.
