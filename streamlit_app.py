import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions


# -----------------------------
# Chroma: connect + retrieve
# -----------------------------
@st.cache_resource
def get_chroma_collection(persist_dir: str, collection_name: str, embedding_model: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"description": f"Collection for {collection_name}"},
    )


def retrieve_chunks(collection, query: str, k: int = 40, where: Optional[Dict[str, Any]] = None):
    return collection.query(query_texts=[query], n_results=k, where=where)


# -----------------------------
# Prompt building (Recruitment)
# -----------------------------
def build_context_with_meta(docs: List[str], metas: List[Dict[str, Any]], max_chars: int = 16000) -> str:
    parts = []
    for doc, meta in zip(docs, metas):
        file_name = meta.get("file_name", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")
        parts.append(f'[file_name="{file_name}" chunk_id={chunk_id}]\n{doc}')
    joined = "\n\n---\n\n".join(parts)
    return joined[:max_chars]


def build_recruitment_messages(
    context: str,
    question: str,
    role_title: str,
    must_have: List[str],
    nice_to_have: List[str],
    seniority: str = "any",
    job_notes: str = "",
    top_n: int = 5,
):
    system_prompt = (
        "You are a technical recruitment assistant.\n\n"
        "Goal:\n"
        "Given a job/project requirement and resume evidence retrieved from multiple candidates, "
        "rank the best candidates.\n\n"
        "Rules:\n"
        "- Use ONLY the provided resume evidence (context). Do NOT invent skills, experience, or education.\n"
        "- Every claim MUST be backed by evidence and include file_name + chunk_id.\n"
        "- If something is not found in the evidence, say 'not found'.\n"
        "- Be fair: do not assume years of experience unless explicitly stated.\n"
        "- Output ONLY valid JSON (no markdown, no extra text)."
    )

    user_prompt = f"""
Job requirements:
Role title: {role_title}
Must-have skills: {must_have}
Nice-to-have skills: {nice_to_have}
Seniority: {seniority}
Notes: {job_notes}

Recruitment question:
{question}

Resume evidence (retrieved chunks across multiple candidates):
{context}

Task:
Rank the best candidates for this role.
Return top {top_n} candidates (or fewer if evidence is weak).

Every score and claim MUST cite evidence with file_name and chunk_id.

Return ONLY JSON with this schema:
{{
  "job": {{
    "role_title": "string",
    "must_have": ["string"],
    "nice_to_have": ["string"],
    "seniority": "intern/junior/mid/senior/any",
    "notes": "string"
  }},
  "ranked_candidates": [
    {{
      "file_name": "string",
      "overall_score": 0,
      "score_breakdown": {{
        "skills_match": 0,
        "project_relevance": 0,
        "experience_level": 0,
        "communication_clarity": 0
      }},
      "highlights": ["string"],
      "gaps": ["string"],
      "evidence": [
        {{
          "file_name": "string",
          "chunk_id": 0,
          "snippet": "string",
          "why_it_matters": "string"
        }}
      ],
      "recommendation": "strong_interview / interview / maybe / no",
      "targeted_interview_questions": ["string"]
    }}
  ],
  "final_notes": {{
    "missing_info": ["string"],
    "tie_breakers_used": ["string"]
  }}
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# -----------------------------
# OpenRouter
# -----------------------------
def openrouter_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/llama-3.1-8b-instruct",
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: int = 60,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"raw_text": r.text}
        raise RuntimeError(f"OpenRouter error {r.status_code}\n{json.dumps(err, indent=2)}")

    data = r.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------
# JSON extraction
# -----------------------------
def extract_json(text: str) -> dict:
    """
    Robust JSON extractor:
    1. Try direct json.loads
    2. Extract the largest {...} block
    3. Retry after cleanup
    """
    if not text:
        raise ValueError("Empty LLM output")

    text = text.strip()

    # 1) Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Extract first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    candidate = match.group(0)

    # 3) Cleanup common issues
    candidate = candidate.replace("“", '"').replace("”", '"')
    candidate = candidate.replace("‘", "'").replace("’", "'")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Show helpful debug info
        raise ValueError(
            f"JSON parsing failed after extraction.\n"
            f"Error: {e}\n\n"
            f"Extracted JSON:\n{candidate}"
        )

# -----------------------------
# Optional: load file_name list (for filter)
# -----------------------------
def load_file_names(cvs_json_path: str) -> List[str]:
    p = Path(cvs_json_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return sorted({d.get("file_name") for d in docs if d.get("file_name")})


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def repair_json_with_llm(bad_json_text: str, model: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You fix JSON. Return ONLY valid JSON. No explanations.",
        },
        {
            "role": "user",
            "content": f"Fix this JSON and return only valid JSON:\n\n{bad_json_text}",
        },
    ]
    return openrouter_chat_completion(
        messages=messages,
        model=model,
        temperature=0,
        max_tokens=800,
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Recruitment RAG (Functional)", layout="wide")
st.title("Recruitment RAG – Candidate Ranking (Functional Streamlit)")

with st.sidebar:
    st.header("Chroma Settings")
    persist_dir = st.text_input("persist_directory", value="./chroma_db")
    collection_name = st.text_input("collection_name", value="paragraph_chunking")
    embedding_model = st.text_input("embedding_model", value="all-mpnet-base-v2")

    st.divider()
    st.header("Retrieval")
    k = st.slider("Top K chunks", 5, 120, 40, 5)
    max_chars = st.slider("Max context chars", 4000, 24000, 16000, 1000)

    st.divider()
    st.header("LLM (locked cheap)")
    model = st.text_input("OpenRouter model", value="meta-llama/llama-3.1-8b-instruct")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("max_tokens", 200, 2000, 900, 50)

    st.divider()
    st.header("Optional CV filter")
    cvs_json_path = st.text_input("CVs.json path", value="data/CVs.json")
    use_filter = st.checkbox("Filter retrieval by one file_name", value=False)

    st.divider()
    top_n = st.slider("Return top N candidates", 1, 10, 5, 1)


# Job inputs
st.subheader("Job Requirements")

col1, col2 = st.columns(2)
with col1:
    role_title = st.text_input("Role title", value="Backend Developer")
    seniority = st.selectbox("Seniority", ["any", "intern", "junior", "mid", "senior"], index=0)
with col2:
    must_have_str = st.text_input("Must-have (comma-separated)", value="Python, Django, REST APIs")
    nice_to_have_str = st.text_input("Nice-to-have (comma-separated)", value="PostgreSQL, Docker, CI/CD")

job_notes = st.text_area("Job notes (optional)", value="Prefer real projects and clean API design.", height=80)

st.subheader("Recruitment Question")
question = st.text_input("Question", value="Who is the best candidate for this backend role?")

must_have = parse_csv_list(must_have_str)
nice_to_have = parse_csv_list(nice_to_have_str)

# Filter selection
where = None
if use_filter:
    file_names = load_file_names(cvs_json_path)
    if file_names:
        selected = st.selectbox("file_name", file_names)
        where = {"file_name": selected}
    else:
        st.warning("No file names found. Check CVs.json path.")
        where = None

# Run
st.divider()
run = st.button("✨ Rank Candidates", type="primary")

if run:
    if not must_have:
        st.error("Must-have list is empty. Add at least one skill.")
        st.stop()

    try:
        collection = get_chroma_collection(persist_dir, collection_name, embedding_model)
        results = retrieve_chunks(collection, question, k=k, where=where)

        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []
        dists = results.get("distances", [[]])[0] if results else []

        context = build_context_with_meta(docs, metas, max_chars=max_chars)
        messages = build_recruitment_messages(
            context=context,
            question=question,
            role_title=role_title,
            must_have=must_have,
            nice_to_have=nice_to_have,
            seniority=seniority,
            job_notes=job_notes,
            top_n=top_n,
        )

        raw = openrouter_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            parsed = extract_json(raw)
        except Exception:
            repaired = repair_json_with_llm(raw, model)
            parsed = extract_json(repaired)

        # attach retrieval trace
        parsed["_retrieval_trace"] = {
            "k": k,
            "query": question,
            "sources": sorted({m.get("file_name") for m in metas if m.get("file_name")}),
        }

        st.success("Done ✅")
        st.subheader("Ranked candidates (JSON)")
        st.json(parsed)

        st.subheader("Retrieved chunks preview")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            st.markdown(f"### Rank {i}  | distance: `{dist:.4f}`")
            st.caption(f'file_name: {meta.get("file_name")} | chunk_id: {meta.get("chunk_id")}')
            st.write(doc)
            st.divider()

    except Exception as e:
        st.error("Failed while parsing model output.")
        st.code(str(e))

        st.subheader("Raw LLM output (debug)")
        st.text_area(
            "Raw response",
            value=raw if "raw" in locals() else "No raw output",
            height=300,
        )

        st.info(
            "Check:\n"
            "- OPENROUTER_API_KEY is set\n"
            "- chroma_db path exists\n"
            "- collection name is correct\n"
        )
