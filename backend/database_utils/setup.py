from typing import List

import pandas as pd
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from tqdm.auto import tqdm

from backend.database_utils.vectore_store_manager import VectorStoreManager


def scrape_documents(wiki_links: List[str]) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=wiki_links,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_="mw-parser-output")
        ),

    )

    docs = []
    for doc in tqdm(loader.lazy_load(), "Scraping documents"):
        docs.append(doc)

    return docs


def setup_vector_store_manager(path_to_documents: str):
    manager = VectorStoreManager(clean=True)
    # get documents
    wiki_links = pd.read_csv(path_to_documents, header=None)[0].tolist()

    # chunk and add new docs
    docs = scrape_documents(wiki_links)
    manager.add_documents(docs)