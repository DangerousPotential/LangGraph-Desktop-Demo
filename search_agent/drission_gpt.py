from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph, add_messages
from DrissionPage import Chromium
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Type
from langgraph.prebuilt import ToolNode, tools_condition


# Define State with a Task Completion Flag
class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_completed: bool  # Flag to track task completion


# Tools Input with DrissionPage
class DrissionPageScreenshotInput(BaseModel):
    """Input Schema for Tool"""
    page_url: str = Field(
        default='https://carousell.sg', 
        description="URL of the page to navigate before taking a screenshot"
    )
    full_page: bool = Field(
        default=False, 
        description="If set to True, it captures the entire page. False captures visible portion only"
    )
    as_base64: bool = Field(
        default=True, 
        description="Return the screenshot as a base64 encoded string"
    )


# Tool Definition with DrissionPage
class DrissionPageScreenshotTool(BaseTool):
    name: str = "Screenshot_Tool"
    description: str = "Takes a screenshot of a webpage and returns an encoded base64 string"
    args_schema: Type[BaseModel] = DrissionPageScreenshotInput

    def _run(self, page_url: str, as_base64: bool = True):
        # Initialize Chromium and take screenshot
        tab = Chromium().latest_tab
        tab.get(page_url)
        screenshot = tab.get_screenshot(as_base64=as_base64)
        tab.close()

        # Return the result with task completion
        return {
            'messages': [
                ToolMessage(content="Base64 Encoded String here: \n" + screenshot)
            ],
            'task_completed': True  # Indicate that the task is done
        }


# Data Flow for Graph

# Set up Graph Builder
graph_builder = StateGraph(State)

# Feature for Chatbot Node
llm_with_tools = ChatOpenAI(model='gpt-4o-mini', temperature=0.0).bind_tools(
    tools=[DrissionPageScreenshotTool()]
)

# Define Chatbot Node
def chatbot(state: State):
    # Check if the task is complete
    if state.get('task_completed', False):
        return {'messages': state['messages']}  # No further actions, task is done

    # Otherwise, invoke the tools
    return {'messages': [llm_with_tools.invoke(state['messages'])]}


# Tool Nodes
tool_node = ToolNode(tools=[DrissionPageScreenshotTool()])

# Add Chatbot Node
graph_builder.add_node('chatbot', chatbot)

# Add Tool Node
graph_builder.add_node('tools', tool_node)

# Define Conditional Edges
def is_task_completed(state: State):
    # Check the task_completed flag in the state
    return state.get('task_completed', False)

graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,  # Default condition to invoke tools
    skip_condition=is_task_completed  # Skip invocation if task is done
)

# Add Edge
graph_builder.add_edge('tools', 'chatbot')

# Connect the START and END nodes
graph_builder.set_entry_point("chatbot")

# Build the graph
graph = graph_builder.compile()


