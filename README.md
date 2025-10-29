# TES RAG System

This project is a RAG (Retrieval-Augmented Generation) system for The Elder Scrolls series. It uses a Streamlit frontend to interact with a FastAPI backend that serves a LangChain agent.

## How to Run

1.  **Install `uv`:**
    ```bash
    pip install uv
    ```

2.  **Install the dependencies:**
    ```bash
    uv sync
    ```

3.  **Set up the vector store:**
    ```bash
    uv run python scripts/setup_vector_store.py
    ```

4.  **Run the backend:**
    ```bash
    uv run uvicorn backend.app.main:app --reload
    ```

5.  **Run the frontend:**
    ```bash
    uv run streamlit run frontend/app.py
    ```

## Project Structure

```
/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI application
│   ├── agent/
│   ├── database_utils/
│   └── ...
├── frontend/
│   ├── app.py             # Streamlit application
│   └── ...
├── scripts/
│   └── setup_vector_store.py # Script to set up the vector store
├── data/
├── credentials/
├── tes_collection_db/
├── .gitignore
├── README.md
└── pyproject.toml         # Project metadata and dependencies for uv
```
