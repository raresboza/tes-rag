from backend.agent.web_agent import get_agent
from backend.database_utils.setup import setup_vector_store_manager
from credentials.setup import setup_credentials
from langchain_core.messages.human import HumanMessage

def main():
    # TODO: make this step optional
    setup_vector_store_manager('data/wiki_links.csv')
    setup_credentials()

    agent = get_agent()

    inputs = {"messages": [HumanMessage(content="Where is the Thieves Guild located in Skyrim?")]}
    config = {"configurable": {"thread_id": "1"}}

    print(agent.invoke(input=inputs, config=config))
    inputs = {"messages": [HumanMessage(content="Is it safe there?")]}
    print(agent.invoke(input=inputs, config=config))

if __name__ == "__main__":
    main()
