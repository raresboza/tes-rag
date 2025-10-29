from fastapi import FastAPI
from pydantic import BaseModel
from backend.agent.web_agent import get_agent
from langchain_core.messages.human import HumanMessage

app = FastAPI()

class Query(BaseModel):
    question: str
    thread_id: str

@app.post("/ask")
def ask(query: Query):
    agent = get_agent()
    inputs = {"messages": [HumanMessage(content=query.question)]}
    config = {"configurable": {"thread_id": query.thread_id}}
    response = agent.invoke(input=inputs, config=config)
    return {"answer": response['messages'][-1].content}
