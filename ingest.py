import faiss
from rag_utils import (
    load_all_pdfs,
    build_chunks,
    get_embeddings,
    normalize_vectors,
    save_index_and_metadata,
    PDF_DIR,
)

def main():
    print("Loading PDFs...")
    docs = load_all_pdfs(PDF_DIR)
    if not docs:
        print("No PDFs found in data/pdfs")
        return

    print(f"Loaded {len(docs)} PDFs")

    print("Chunking documents...")
    chunks = build_chunks(docs)
    print(f"Created {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    print("Generating embeddings...")
    embeddings = get_embeddings(texts)
    embeddings = normalize_vectors(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving FAISS index and metadata...")
    save_index_and_metadata(index, chunks)

    print("Done.")
    print(f"Indexed {len(chunks)} chunks from {len(docs)} PDFs.")


if __name__ == "__main__":
    main()