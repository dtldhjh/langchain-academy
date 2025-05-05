'''
Description: 
version: 
Author: hjh
Date: 2025-03-23 22:44:52
LastEditors: hjh
LastEditTime: 2025-05-04 21:29:57
'''
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import time
import jwt
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 3600*24*30, # 填写您期望的有效时间
        "nbf": int(time.time()) # 填写您期望的生效时间
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

ak = os.getenv('ak')
sk = os.getenv('sk')

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(api_key=encode_jwt_token(ak, sk),
    base_url="https://api.sensenova.cn/compatible-mode/v1/",
    model_name="SenseChat-5",)
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()