'''
Description: 
version: 
Author: hjh
Date: 2025-03-23 22:44:52
LastEditors: hjh
LastEditTime: 2025-05-04 21:30:36
'''
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

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

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(api_key=encode_jwt_token(ak, sk),
    base_url="https://api.sensenova.cn/compatible-mode/v1/",
    model_name="SenseChat-5",)
llm_with_tools = llm.bind_tools([multiply])
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.如果用户问题是有关算数的，必须调用工具。")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
