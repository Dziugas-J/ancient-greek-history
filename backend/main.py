from system import split_book, create_faiss_index, build_entity_list, retrieve_chunks, generate_answer


def main():
    chunks = split_book()
    index = create_faiss_index(chunks)
    entity_list = build_entity_list(chunks)

    while True:
        query = input("\n> ").strip()
        if not query:
            break

        results = retrieve_chunks(query, index, chunks, entity_list)
        answer = generate_answer(query, results)
        print(answer)


if __name__ == "__main__":
    main()
