import faiss
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BOOK_PATH = "data/book_of_the_ancient_greeks.txt"

model = SentenceTransformer("all-MiniLM-L6-v2")
openai_client = OpenAI()


def load_book():
    with open(BOOK_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    df = pd.DataFrame({"text": lines})
    return df


def split_book():
    with open(BOOK_PATH, encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    return chunks


def create_faiss_index(chunks):
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0]]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
