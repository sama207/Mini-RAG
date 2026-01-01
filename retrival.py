import json
import re
import tiktoken
## same sentence-transformer model for semantic chunking and retrieval,
## ensuring embedding space consistency, while respecting library-specific embedding interfaces.”
# embbiding model for chroma
from chromadb.utils import embedding_functions
# embbiding model for semanticChuncker
from langchain_community.embeddings import HuggingFaceEmbeddings
# lc_embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
from chunking_evaluation.chunking import KamradtModifiedChunker
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

DATA_PATH="data\CVs.json"
def clean_pdf_text(t: str) -> str:
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def prepare_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    
    texts = []
    metadatas = []

    for item in docs:
        text = clean_pdf_text(item["text"])
        texts.append(text)
        metadatas.append({
            "doc_id": item["id"],
            "file_name": item["file_name"],
            "source": item.get("source", "unknown")
        })
        
    return texts,metadatas

def count_tokens(text, model="cl100k_base"):
    """Count tokens in a text string using tiktoken"""
    encoder = tiktoken.get_encoding(model)
    print(f"Number of tokens: {len(encoder.encode(text))}")

def analyze_chunks(chunks, numOfChunk1,numOfChunk2,use_tokens=False):
    # Print the chunks of interest
    print("\nNumber of Chunks:", len(chunks))
    print("\n", "="*50, f"{numOfChunk1} Chunk", "="*50,"\n", chunks[numOfChunk1])
    print("\n", "="*50, f"{numOfChunk2} Chunk", "="*50,"\n", chunks[numOfChunk2])
    
    chunk1, chunk2 = chunks[numOfChunk1], chunks[numOfChunk2]
    
    if use_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens1 = encoding.encode(chunk1)
        tokens2 = encoding.encode(chunk2)
        
        # Find overlapping tokens
        for i in range(len(tokens1), 0, -1):
            if tokens1[-i:] == tokens2[:i]:
                overlap = encoding.decode(tokens1[-i:])
                print("\n", "="*50, f"\nOverlapping text ({i} tokens):", overlap)
                return
        print("\nNo token overlap found")
    else:
        # Find overlapping characters
        for i in range(min(len(chunk1), len(chunk2)), 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                print("\n", "="*50, f"\nOverlapping text ({i} chars):", chunk1[-i:])
                return
        print("\nNo character overlap found")

def split_into_paragraphs(text):
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]

def chunk_paragraphs(paragraphs,token_splitter):
    chunks = []
    for p in paragraphs:
        if len(p.split()) < 120:   # short paragraph → keep
            chunks.append(p)
        else:                      # long paragraph → token split
            chunks.extend(token_splitter.split_text(p))
    return chunks

def main():
    # ========= Prepare Data =========
    texts, metadatas = prepare_data(DATA_PATH)
    
    # Embedding function for Chroma
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    # =========== Semantic Chunking =========== 
    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=400,  # Target size in tokens
        min_chunk_size=50,  # Initial split size
        embedding_function=embedding_fn,  # Pass your embedding function
    )

    modified_kamradt_chunks = []
    for text, meta in zip(texts, metadatas):
        # Split text
        chunks = kamradt_chunker.split_text(text)  # <-- one CV at a time
        
        for i, ch in enumerate(chunks):
            modified_kamradt_chunks.append(ch)

    print("Semantic chunking \n\n")
    analyze_chunks(modified_kamradt_chunks,3,6 ,use_tokens=True)
    print("\n\n", "="*50, "\n\n")
    count_tokens(modified_kamradt_chunks[3])
    count_tokens(modified_kamradt_chunks[6])


    # ============ Paragraph chunking ============
    # Initialize the splitter
    token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=200, chunk_overlap=30)

    paragraph_chunks = []
    paragraph_chunk_metas = []
    paragraph_chunk_ids = []

    for text, meta in zip(texts, metadatas):
        paragraphs = split_into_paragraphs(text)
        chunks = chunk_paragraphs(paragraphs, token_splitter)

        for i, ch in enumerate(chunks):
            paragraph_chunks.append(ch)
            paragraph_chunk_metas.append({**meta, "chunk_id": i, "chunk_type": "paragraph"})
            paragraph_chunk_ids.append(f"{meta['doc_id']}_par_{i}")
            
    print("paragraph chunking \n\n")
    print("Total paragraph chunks:", len(paragraph_chunks))

    analyze_chunks(paragraph_chunks, 3, 6, use_tokens=True)

    print("Tokens chunk[3]:", count_tokens(paragraph_chunks[3]))
    print("Tokens chunk[6]:", count_tokens(paragraph_chunks[6]))

if __name__ == "__main__":
    main()
