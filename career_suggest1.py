#imports
import chromadb
from chromadb.utils import embedding_functions
import json
from pprint import pprint

def build_careers_index():
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name="careers",
        embedding_function=embedding_fn
    )

    with open("data/careers.json", "r", encoding="utf-8") as f:
        careers = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for c in careers:
        text_block = f"""
Title: {c['title']}
Category: {c['category']}
Seniority: {c['seniority']}

Summary: {c['summary']}

Ideal background: {c['ideal_background']}

Core skills: {', '.join(c['core_skills'])}
Nice to have skills: {', '.join(c['nice_to_have_skills'])}

Interests fit: {', '.join(c['interests_fit'])}
Personality fit: {', '.join(c['personality_fit'])}

Typical tasks: {', '.join(c['typical_tasks'])}

Common job titles: {', '.join(c['common_job_titles'])}

Suggested learning paths: {', '.join(c['suggested_learning_paths'])}

Tags: {', '.join(c['tags'])}
"""
        documents.append(text_block)
        metadatas.append({
            "id": c["id"],
            "title": c["title"],
            "category": c["category"],
            "seniority": c["seniority"]
        })
        ids.append(c["id"])

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Indexed {len(documents)} careers into Chroma.")

def get_careers_collection():
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name="careers",
        embedding_function=embedding_fn
    )

    return collection

def retrieve_careers(profile_text: str, k: int = 5):
    """
    Takes a text describing the candidate (resume text or summary)
    and returns top-k relevant careers from Chroma.
    """
    collection = get_careers_collection()

    results = collection.query(
        query_texts=[profile_text],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    career_results = []
    for doc, meta, _id in zip(docs, metas, ids):
        career_results.append({
            "id": _id,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "seniority": meta.get("seniority"),
            "raw_text": doc
        })

    return career_results


# Example usage
if __name__ == "__main__":
    # 🔹 Only run this the first time or when careers.json changes
    # build_careers_index()

    profile_text = """
    Computer science student with strong skills in Python, machine learning, and data visualization.
    Experience in building ML models, working with pandas and scikit-learn, and creating dashboards.
    Interested in AI, data science, and solving analytical problems.
    """

    results = retrieve_careers(profile_text, k=5)
    pprint(results)
