#imports
import chromadb
from chromadb.utils import embedding_functions
import json
from pprint import pprint
from pypdf import PdfReader
import os
from typing import List, Dict

def read_pdf(path: str) -> str:
    """
    Read PDF file and return its full text
    """
    reader = PdfReader(path)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text 

    return text

def build_pdf_corpus_json(input_dir: str, output_json_path: str) -> None:
    """
    Read all PDF files from input_dir, extract their text, and save them as a JSON list.
    Each entry will look like:
    {
        "id": "resume_0",
        "file_name": "samaShalabiCV(AI).pdf",
        "text": "full extracted text...",
        "source": "resume"
    }
    """
    data: List[Dict] = []
    idx = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue  # skip non-PDFs

        pdf_path = os.path.join(input_dir, filename)
        print(f"Reading PDF: {pdf_path}")

        text = read_pdf(pdf_path)

        doc = {
            "id": f"doc_{idx}",
            "file_name": filename,
            "text": text,
            "source": "pdf"
        }
        data.append(doc)
        idx += 1

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} PDF documents into JSON: {output_json_path}")


def build_pdfs_index_from_json(json_path: str, collection_name: str = "pdf_docs"):
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for d in docs:
        text = d["text"]
        doc_id = d["id"]

        documents.append(text)
        metadatas.append({
            "id": d["id"],
            "file_name": d["file_name"],
            "source": d.get("source", "pdf")
        })
        ids.append(doc_id)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Indexed {len(documents)} careers into Chroma.")

def get_pdf_collection(collection_name: str = "pdf_docs"):
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection

def retrieve_from_pdfs(query_text: str, k: int = 5, collection_name: str = "pdf_docs"):
    """
    Retrieve top-k PDF documents (or chunks) relevant to the query_text.
    """
    collection = get_pdf_collection(collection_name)

    results = collection.query(
        query_texts=[query_text],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    out = []
    for doc, meta, _id in zip(docs, metas, ids):
        out.append({
            "id": _id,
            "file_name": meta.get("file_name"),
            "source": meta.get("source"),
            "text": doc
        })

    return out

if __name__ == "__main__":
    # # 1) First, build the JSON corpus
    # build_pdf_corpus_json(
    #     input_dir="data/resumes",
    #     output_json_path="data/careers.json"
    # )

    # # 2) Then, index that corpus into Chroma
    # build_pdfs_index_from_json(
    #     json_path="data/careers.json",
    #     collection_name="pdf_docs"
    # )

    query = "Python and backend skills with web development experience"
    results = retrieve_from_pdfs(query, k=3)
    pprint(results)