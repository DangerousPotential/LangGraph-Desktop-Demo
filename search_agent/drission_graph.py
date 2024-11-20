from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph, add_messages
from DrissionPage import Chromium
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Type
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tools Input with DrissionPage
class DrissionPageScreenshotInput(BaseModel):
    """Input Schema for Tool"""
    page_url: str = Field(default='https://carousell.sg', description= "URL of the page to navigate before taking a screenshot")
    as_base64: bool = Field(default = True, description="Return the screenshot as a base64 encoded string")
# Tool Definition with DrissionPage
class DrissionPageScreenshotTool(BaseTool):
    name: str = "Screenshot_Tool"
    description: str = (
        "Takes a screenshot of a webpage and returns encoded base64 string"
    )
    args_schema: Type[BaseModel] = DrissionPageScreenshotInput

    def _run(self, page_url: str, as_base64: bool = True):
        from DrissionPage import Chromium

        tab = Chromium().latest_tab
        tab.get(page_url)
        screenshot = tab.get_screenshot(as_base64 = True)
        tab.close()
        return {'messages': [llm_with_tools.invoke([HumanMessage(content = [
            {'type': 'text', 'text': 'What is written in the webpage?'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64, {screenshot}'}}
        ])])]}



# Data Flow for Graph

# Set up Graph Builder
graph_builder = StateGraph(State)

# Feature for Chatbot Node
llm_with_tools = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.0).bind_tools(tools = [DrissionPageScreenshotTool()])

# Define Chatbot Node
def chatbot(state: State):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

# Tool Nodes
tool_node = ToolNode(tools = [DrissionPageScreenshotTool()])

# Add Chatbot Node
graph_builder.add_node('chatbot', chatbot)

# Add Tool Node
graph_builder.add_node('tools', tool_node)

# Define Connections
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)

# Add Edge
graph_builder.add_edge('tools', 'chatbot')

# Connect the START AND END
graph_builder.set_entry_point("chatbot")

# build the graph
graph = graph_builder.compile()




