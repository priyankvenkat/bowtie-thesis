import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Helper function to load and preprocess chunks
def load_chunks(filepath, source_name):
    """
    Load chunks from JSON and prepare text strings for embedding.

    Args:
        filepath (str): Path to the JSON chunk file.
        source_name (str): Source identifier for chunk IDs.

    Returns:
        texts (List[str]): List of text strings extracted from chunks for embedding.
        processed (List[Dict]): List of metadata dicts for each chunk.
    """
    with open(filepath, "r") as f:
        chunks = json.load(f)

    processed = []
    texts = []

    for idx, chunk in enumerate(chunks):
        chunk_type = chunk.get("type")
        page = chunk.get("page")
        content = ""

        if chunk_type == "text":
            content = chunk.get("content", "")
        elif chunk_type == "table":
            title = chunk.get("table_title", "")
            columns = " ".join(col if col is not None else "" for col in chunk.get("columns", []))
            rows = [" ".join(cell if cell is not None else "" for cell in row) for row in chunk.get("rows", [])]
            row_text = " ".join(rows[:3])  # limit rows for embedding
            content = f"{title} {columns} {row_text}"

        texts.append(content)
        processed.append({
            "id": f"{source_name}_{idx}",
            "page": page,
            "type": chunk_type,
            "source": source_name,
            "chunk": chunk
        })

    return texts, processed

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load all three sets
sources = {
    "tables": "/mnt/ssd850/Bowties/single_llm_with_api/tables.json",
    "sections": "/mnt/ssd850/Bowties/single_llm_with_api/sections.json",
    "combined": "/mnt/ssd850/Bowties/single_llm_with_api/combined_chunks.json"
}

# Process each source
for source_name, filepath in sources.items():
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        continue

    print(f"🔄 Processing {filepath}...")
    texts, metadata = load_chunks(filepath, source_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build and save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, f"faiss_{source_name}.idx")
    with open(f"faiss_{source_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved index and metadata for '{source_name}' with {len(texts)} chunks\n")
