from langchain.document_loaders.parsers.html import bs4

from model.gemini import call_gemini
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Annotated
from typing_extensions import TypedDict
import os

def read_credentials(file_path):
    """Read API key from text file"""
    with open(file_path, 'r') as file:
        return file.read().strip()

def scrape_documents() -> List[Document]:
    wiki_links = [
            "https://en.uesp.net/wiki/Lore:Solitude",
            "https://en.uesp.net/wiki/Lore:Falkreath",
            "https://en.uesp.net/wiki/Lore:Daedra",
            "https://en.uesp.net/wiki/Lore:Aedra",
            "https://en.uesp.net/wiki/Lore:Riverwood",
            "https://en.uesp.net/wiki/Lore:Ivarstead",
            "https://en.uesp.net/wiki/Lore:Rorikstead",
            "https://en.uesp.net/wiki/Lore:Morthal",
            "https://en.uesp.net/wiki/Lore:Dawnstar",
            "https://en.uesp.net/wiki/Lore:Whiterun"
            "https://en.uesp.net/wiki/Lore:Windhelm",
            "https://en.uesp.net/wiki/Lore:Winterhold",
            "https://en.uesp.net/wiki/Lore:Helgen",
            "https://en.uesp.net/wiki/Lore:Markarth",
            "https://en.uesp.net/wiki/Lore:Mephala",
            "https://en.uesp.net/wiki/Lore:Riften",
        ]

    loader = WebBaseLoader(
        web_paths=wiki_links,
        bs_kwargs=dict(
            parseonly=bs4.SoupStrainer(class_="mw-parser-output")
        ),
        show_progress=True
    )

    docs = loader.load()
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vector_store = Chroma(
        collection_name="tes_collection",
        embedding_function=embeddings,
        persist_directory="./tes_collection_db",
    )
    return vector_store

def embed_documents(vector_store, docs: List[Document]):
    vector_store.add_documents(documents=docs)

def ingestion_pipeline():
    docs = scrape_documents()
    splits = chunk_documents(docs)
    vector_store = get_vector_store()
    embed_documents(vector_store, splits)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def retrieve(state: State, vector_store):
    query = state['messages'][-1].text()
    print(query)
    search_results = vector_store.similarity_search(query, k=5)
    docs_content = f"Documents retrieved for {query}" + "\n\n".join(doc.page_content for doc in search_results)
    return {"messages": [ToolMessage(content=docs_content, tool_call_id="tool_id")]}


def answer(state: State, model):
    prompt = """
    You are an expert in Elder Scrolls lore. You are linked to a lore wikipedia database.
    Your role is to answer the questions of the user using the information from the database.
    Make sure to only answer the user queries with the information the database.
    Make sure to cite which information from the database you have used.
    Please find the conversation below.
    """
    conversation_messages = [SystemMessage(content=prompt)] + state['messages']
    response = model.invoke(conversation_messages)
    return {"messages": [response]}



def get_lang_graph():
    vector_store = get_vector_store()
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    graph = StateGraph(State)
    graph.add_node("AnswerNode", lambda state: answer(state, llm))
    graph.add_node("RetrievalNode", lambda state: retrieve(state, vector_store))
    graph.add_edge("__start__", "RetrievalNode")
    graph.add_edge("RetrievalNode", "AnswerNode")
    graph.add_edge("AnswerNode", "__end__")
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    return compiled_graph


def main():
    api_key = read_credentials("credentials/google_api_key.txt")
    os.environ["GOOGLE_API_KEY"] = api_key
    graph = get_lang_graph()

    inputs = {"messages": [HumanMessage(content="Where is the Thieves Guild located in Skyrim?")]}
    config = {"configurable": {"thread_id": "1"}}

    print(graph.invoke(input=inputs, config=config))
    inputs = {"messages": [HumanMessage(content="Is it safe there?")]}
    print(graph.invoke(input=inputs, config=config))
    # vector_store = get_vector_store()
    # query="Where is the Thieves Guild located in Skyrim?"
    # search_results = vector_store.similarity_search_with_score(query, k=5)

    # for result, score in search_results:
    #     print("========================")
    #     print(result.page_content)
    #     print("------------------------")
    #     print(result.metadata)
    #     print(f"score: {score}")
    #     print("========================")


if __name__ == "__main__":
    main()
