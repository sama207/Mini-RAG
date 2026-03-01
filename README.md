# Recruitment RAG System -- Resume Search & Evaluation

This project implements a full Retrieval-Augmented Generation (RAG)
pipeline for resume search, chunking evaluation, indexing with ChromaDB,
and an interactive Streamlit recruitment chatbot.

The system includes:

-   Resume preprocessing
-   Paragraph-based chunking pipeline
-   ChromaDB vector indexing
-   Retrieval evaluation utilities
-   Streamlit recruitment chatbot UI
-   OpenRouter LLM integration

------------------------------------------------------------------------

# 📁 Project Structure

    main.py                  → Builds & indexes resume chunks into ChromaDB
    streamlit_app.py         → Recruitment RAG chatbot (UI)
    evaluate_chunking.py     → Evaluation utilities
    src/                     → Core pipeline (chunkers, indexers, preprocessors)
    data/                    → CVs + QA + queries
    chroma_db/               → Persistent ChromaDB (already included)

------------------------------------------------------------------------

# 🧠 How The System Works

1.  CVs are loaded from `data/CVs.json`
2.  They are chunked using the ParagraphChunker
3.  Chunks are embedded and indexed into ChromaDB
4.  The Streamlit app retrieves relevant chunks
5.  The LLM answers using ONLY retrieved resume evidence

------------------------------------------------------------------------

# 🛠 Requirements

-   Python 3.9+
-   OpenRouter API Key

------------------------------------------------------------------------

# 📦 Installation

From the project root:

``` bash
python -m venv venv
```

Activate:

**Windows**

``` bash
venv\Scripts\activate
```

**Mac/Linux**

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔐 Environment Variable

Create a `.env` file in the root:

    OPENROUTER_API_KEY=your_openrouter_key_here

This is required for the Streamlit chatbot.

------------------------------------------------------------------------

# ⚙️ IMPORTANT: How To Run The Project Correctly

There are TWO possible scenarios:

------------------------------------------------------------------------

## ✅ Scenario 1 --- Use Existing Indexed Database (Fastest)

The project already includes a prebuilt `chroma_db/` folder.

You can directly launch the chatbot:

``` bash
streamlit run streamlit_app.py
```

Open:

    http://localhost:8501

No indexing needed.

------------------------------------------------------------------------

## 🔄 Scenario 2 --- Rebuild The Index (Optional)

If you want to rebuild embeddings and re-index resumes:

``` bash
python main.py
```

This will: - Load resumes from `data/CVs.json` - Chunk them - Create a
ChromaDB collection (`paragraph_chunking`) - Store everything inside
`./chroma_db`

After indexing completes, launch the chatbot:

``` bash
streamlit run streamlit_app.py
```

------------------------------------------------------------------------

# 🧪 Testing The Retrieval (Optional)

`main.py` contains a search example inside the script.

You can uncomment:

``` python
# search_example()
```

to test retrieval from terminal.

------------------------------------------------------------------------

# 🎛 Streamlit Sidebar Settings

When running the chatbot:

-   persist_directory → `./chroma_db`
-   collection_name → `paragraph_chunking`
-   embedding_model → `all-mpnet-base-v2` (must match indexing model)
-   OpenRouter model → e.g. `openai/gpt-5-mini`

⚠️ If embedding_model does not match the one used during indexing,
retrieval may fail.

------------------------------------------------------------------------

# 🛡 Grounding Rules

The chatbot:

-   Uses ONLY retrieved resume chunks
-   Must cite (file_name, chunk_id)
-   Does NOT hallucinate missing experience
-   Returns "not found in the provided resumes" when evidence is missing

------------------------------------------------------------------------

# 🧩 Troubleshooting

### ❌ Chroma Collection Not Found

Run:

``` bash
python main.py
```

### ❌ Missing OPENROUTER_API_KEY

Ensure `.env` exists in root folder.

### ❌ Wrong Embedding Model

Make sure Streamlit embedding_model matches the one used in `main.py`.

------------------------------------------------------------------------

# 🎓 Purpose

This project demonstrates:

-   Chunking strategy comparison
-   Retrieval evaluation metrics
-   Production-style RAG architecture
-   Resume-based candidate ranking system

------------------------------------------------------------------------

# 🚀 Final Run Command (Correct Order)

If database already exists:

``` bash
streamlit run streamlit_app.py
```

If rebuilding from scratch:

``` bash
python main.py
streamlit run streamlit_app.py
```

Always run commands from the project root directory.
