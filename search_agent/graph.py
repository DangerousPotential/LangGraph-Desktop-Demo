from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Chat Feature for Node
llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.0)

# Create Node
def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}

# Add Node to Graph
graph_builder.add_node('chatbot', chatbot)

# Define Connections
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

# Compile Graph
graph = graph_builder.compile()

# View Graph
graph.get_graph().draw_mermaid_png(output_file_path= './graph.png')


