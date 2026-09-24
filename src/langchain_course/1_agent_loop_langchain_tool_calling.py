from optparse import Values
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# tools langchain tool decorator

@tool
def get_product_price(product: str) -> float:
    """Look Up the price of a product in the catalog..."""
    print(f" >>Executin the grt product fn ")
    prices = {"laptop":1333.99 , "headphones" : 1000, "keyboard": 2000}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to price
    Available tiers: bronze, silver, gold"""
    print(f" >> Executing the apply_discount(price={price}, '{discount_tier}')")
    discount_percentages = {"bronze":5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Agent loop

@traceable(name = "Langchain agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}" , temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question:{question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content = (
                "you are  a helpful shopping assistant"
                "you've the access to prod catalog"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly"
                "1. Never guess any product price"
                "2. Only call apply_discount AFTER you've received"
                "a price from get_product_price. pass the exact price"
                "returned by get_product_price - don't pass a made up number. \n"
                "3.Never calculate discounts on your own math"
                "Always use apply_discount_tool"
            )
        ),
        HumanMessage(content= question)
    ]
    for iteration in range(1, MAX_ITERATIONS +1):
        print(f"\n ---iteration {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #if not tool calls, ai_message.content is answer
        if not tool_calls:
            print(f"Final Answer {ai_message.content}")
            return ai_message.content
        
        #process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content = str(observation), tool_call_id = tool_call_id)
        )
    print("ERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent(.bind_tools)!")
    print()
    result = run_agent("whats the price of a laptop after gold discount?")