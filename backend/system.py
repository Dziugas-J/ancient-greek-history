from pathlib import Path

import faiss
import spacy
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

load_dotenv()

BOOK_PATH = Path(__file__).resolve().parent / "data" / "book_of_the_ancient_greeks.txt"
ENTITY_LABELS = {"PERSON", "GPE", "LOC", "EVENT"}
FUZZY_MATCH_THRESHOLD = 85

model = SentenceTransformer("all-MiniLM-L6-v2")
openai_client = OpenAI()
nlp = spacy.load("en_core_web_sm")


def split_book():
    with open(BOOK_PATH, encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    return chunks


def create_faiss_index(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def build_entity_list(chunks):
    entities = set()

    for doc in nlp.pipe(chunks):
        for ent in doc.ents:
            text = " ".join(ent.text.split())
            if ent.label_ in ENTITY_LABELS and len(text) >= 3:
                entities.add(text)

    return list(entities)


def correct_query(query, entity_list):
    corrections = {}
    corrected_words = []

    for word in query.split():
        stripped = word.strip(".,!?;:'\"")
        if not stripped:
            corrected_words.append(word)
            continue

        match = process.extractOne(stripped, entity_list, scorer=fuzz.ratio, processor=str.lower)

        if match and match[1] >= FUZZY_MATCH_THRESHOLD and match[0].lower() != stripped.lower():
            corrections[stripped] = match[0]
            corrected_words.append(word.replace(stripped, match[0]))
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), corrections


def retrieve_chunks(query, index, chunks, entity_list, k=5):
    corrected_query, _ = correct_query(query, entity_list)

    query_embedding = model.encode(
        [corrected_query],
        normalize_embeddings=True
    )

    _, indices = index.search(
        query_embedding,
        k
    )

    results = [chunks[idx] for idx in indices[0]]

    return results


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
