import re
import inspect
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
    price = float(price)
    print(f" >> Executing the apply_discount(price={price}, '{discount_tier}')")
    discount_percentages = {"bronze":5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions=[]
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function,"__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names =", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model,messages = messages, options=options)

# Agent loop

@traceable(name = "Ollama agent Loop")
def run_agent(question: str):
    print(f"Question:{question}")
    print("=" * 60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = "" 

    for iteration in range(1, MAX_ITERATIONS +1):
        print(f"\n ---iteration {iteration}")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages = [{"role":"user", "content":full_prompt}],
            options={"stop":["\nObservation"],"temparature":0}
        )
        output = response.message.content

        print(f"LLM OUTPUT \n{output}")

        print(f" [Parsing] Looking for Final Answer")
        final_answer_match = re.match(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer:{final_answer}")
            print("\n" + "=" *60)
            print(f"Final Answer: {final_answer}")
            return final_answer


        #CHNANGE 6: parse tool calls from raw text with regex - fragile if LLM doesnt follow format
        print(f" [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("[Parsing] ERROR: Could not parse Action/Action Input from LLM Output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f"[Tool Selected] {tool_name} with args: {tool_input_raw}")

         # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent(.bind_tools)!")
    print()
    result = run_agent("whats the price of a laptop after gold discount?")