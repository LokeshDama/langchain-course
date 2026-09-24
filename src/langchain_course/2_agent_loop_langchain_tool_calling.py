
from itertools import product
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
import ollama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# tools langchain tool decorator

traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look Up the price of a product in the catalog..."""
    print(f" >>Executin the grt product fn ")
    prices = {"laptop":1333.99 , "headphones" : 1000, "keyboard": 2000}
    return prices.get(product, 0)

traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to price
    Available tiers: bronze, silver, gold"""
    print(f" >> Executing the apply_discount(price={price}, '{discount_tier}')")
    discount_percentages = {"bronze":5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# without langchain tool decorator we need to manually define the json schema based on the modal 
# we're using

tools_for_llm = [
    {
        "type":"function",
        "function": {
            "name":"get_product_price",
            "description":"Look Up the price of a product in the catalogue",
            "parameters":{
                "product": {
                    "type":"string",
                    "description":"the product name, e.g. 'laptop','headphones', 'keyboard' "
                },
            },
            "required":["product"]
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a given price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the item",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages = messages)

# Agent loop

@traceable(name = "Ollama agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question:{question}")
    print("=" * 60)

    messages = [
        { 
            "role":"system",
            "content" : (
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
        },
        {"role":"user", "content": question}
    ]
    for iteration in range(1, MAX_ITERATIONS +1):
        print(f"\n ---iteration {iteration}")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        #if not tool calls, ai_message.content is answer
        if not tool_calls:
            print(f"Final Answer {ai_message.content}")
            return ai_message.content
        
        #process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use(**tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
          {
            "role":"tool",
            "content": str(observation)
          }
        )
    print("ERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent(.bind_tools)!")
    print()
    result = run_agent("whats the price of a laptop after gold discount?")