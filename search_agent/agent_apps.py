from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph, add_messages
from DrissionPage import Chromium
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Type
from langgraph.prebuilt import create_react_agent



# Tools Input with DrissionPage
class DrissionPageScreenshotInput(BaseModel):
    """Input Schema for Tool"""
    page_url: str = Field(default='https://carousell.sg', description= "URL of the page to navigate before taking a screenshot")
    full_page: bool = Field(default = False, description = "If set to True, it captures the entire page. False are visible portion only")
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
        tab.get('https://carousell.sg')
        screenshot = tab.get_screenshot(as_base64 = True)
        tab.close()
        return screenshot


tools = [DrissionPageScreenshotTool()]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)
model = ChatOpenAI(model = 'gpt-4o-mini', temperature= 0.0)
agent = create_react_agent(model, tools)


print(agent.invoke({'messages': 'go to https://carousell.sg and take a screenshot and return to me the base64 string'}))
