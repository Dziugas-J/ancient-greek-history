# Ancient Greek History RAG

A Retrieval-Augmented Generation system over a public-domain book on ancient
Greek history ([Project Gutenberg #68180](https://www.gutenberg.org/ebooks/68180)),
built incrementally.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Stage 1 — Step 1: Load the book

The book text lives at `data/book_of_the_ancient_greeks.txt`.

Verify loading works:
```
python backend\loader.py
```

(Further steps — chunking, embeddings, retrieval, generation — will be added
incrementally.)
