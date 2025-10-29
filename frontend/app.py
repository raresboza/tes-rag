import streamlit as st
import requests

st.title("TES RAG System")

question = st.text_input("Ask a question about The Elder Scrolls:")

if st.button("Submit"):
    if question:
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://127.0.0.1:8000/ask", json={"question": question, "thread_id": "1"})
                response.raise_for_status()  # Raise an exception for bad status codes
                answer = response.json().get("answer", "No answer found.")
                st.write(answer)
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the backend: {e}")
    else:
        st.warning("Please enter a question.")
