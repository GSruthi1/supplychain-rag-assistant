from rag_utils import search_similar_chunks, generate_answer

def main():
    print("\nRAG PDF Demo")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        print("\nRetrieving relevant chunks...\n")
        chunks = search_similar_chunks(question, top_k=5)

        if not chunks:
            print("No relevant chunks found.\n")
            continue

        print("Top retrieved chunks:")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk["text"][:180].replace("\n", " ")
            print(f"{i}. {chunk['source']} | chunk {chunk['chunk_index']} | score={chunk['score']:.4f}")
            print(f"   {preview}...\n")

        print("Generating final answer...\n")
        answer = generate_answer(question, chunks)

        print("Answer:")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()