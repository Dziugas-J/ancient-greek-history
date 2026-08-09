from system import load_book, split_book, create_faiss_index, retrieve_chunks, generate_answer


def main():
    df = load_book()
    print(df.shape)

    chunks = split_book()
    print(len(chunks))

    index = create_faiss_index(chunks)
    print(index.ntotal)

    query = "Who was Pericles?"
    results = retrieve_chunks(query, index, chunks)

    answer = generate_answer(query, results)
    print(answer)


if __name__ == "__main__":
    main()
