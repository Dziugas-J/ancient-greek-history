# Ancient Greek History RAG

Built a retrieval-augmented QA system over a full-length book on ancient Greek history. System chunks and embeds the text with sentence-transformers, retrieves via FAISS cosine similarity, and corrects misspelled names/places in queries using spaCy NER + fuzzy matching before generating answers with GPT-4o-mini. Demo is built on StreamLit UI and Python as backend.

System also shows the retrieved chunk after sending the answer.

Some questions to test the system on the demo below:

## Positives:
Who was Pericles?
Tell me about the Trojan War.
Who was Achiles?
What was Sparta known for?
Who was Socrates?

## Negatives:
Who was AchhilllleSes? (correction won't trigger, retrieval will be weak/irrelevant)
What's the capital of France? (tests it doesn't hallucinate an answer from general knowledge)
What did Julius Caesar do in Rome? (real historical figure, but likely outside this book's scope)
asdkjaslkdj Sparta wjkefj (pure gibberish)
What will happen in ancient Greece next year?

# Demo
https://ancient-greek-history-fknr7diydzjudjrxhwltfj.streamlit.app/
