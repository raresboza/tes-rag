from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph, START, END

from backend.agent.llm_prompts import tes_llm_agent_prompt
from backend.database_utils.vectore_store_manager import VectorStoreManager


class State(TypedDict):
    messages: Annotated[list, add_messages]


def retrieve(state: State, manager: VectorStoreManager):
    query = state['messages'][-1].text()
    print(query)
    search_results = manager.search(query, k=5)
    docs_content = f"Documents retrieved for {query}" + "\n\n".join(doc.page_content for doc in search_results)
    return {"messages": [ToolMessage(content=docs_content, tool_call_id="tool_id")]}


def answer(state: State, model):
    conversation_messages = [SystemMessage(content=tes_llm_agent_prompt)] + state['messages']
    response = model.invoke(conversation_messages)
    return {"messages": [response]}


def get_agent():
    manager = VectorStoreManager()
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    graph = StateGraph(State)
    graph.add_node("AnswerNode", lambda state: answer(state, llm))
    graph.add_node("RetrievalNode", lambda state: retrieve(state, manager))
    graph.add_edge(START, "RetrievalNode")
    graph.add_edge("RetrievalNode", "AnswerNode")
    graph.add_edge("AnswerNode", END)
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    return compiled_graph