import streamlit as st
from system import build_entity_list, create_faiss_index, generate_answer, retrieve_chunks, split_book

st.set_page_config(page_title="Ancient Greek History")
st.title("Ancient Greek History")


@st.cache_resource
def load_system():
    chunks = split_book()
    index = create_faiss_index(chunks)
    entity_list = build_entity_list(chunks)
    return chunks, index, entity_list


chunks, index, entity_list = load_system()

MAX_QUESTIONS = 10

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

query = st.text_input("Ask a question about ancient Greek history")
st.caption(f"{st.session_state.question_count}/{MAX_QUESTIONS} questions used")

if st.button("Ask") and query:
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.warning("You've reached the limit of 10 questions for this demo.")
    else:
        st.session_state.question_count += 1

        with st.spinner("Thinking..."):
            results = retrieve_chunks(query, index, chunks, entity_list)
            answer = generate_answer(query, results)

        st.write(answer)

        with st.expander("Retrieved chunks"):
            for chunk in results:
                st.write(chunk)
