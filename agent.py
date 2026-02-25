from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI 
import os
from dotenv import load_dotenv

# Get full path to .env file
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")

print("Looking for .env at:", env_path)

load_dotenv(dotenv_path=env_path)

print("Loaded key:", os.getenv("OPENAI_API_KEY"))

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # 👈 force pass key
)

def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        format_type, actual_text = text.split(":", 1)
        format_type = format_type.strip().lower()
        actual_text = actual_text.strip()

        if format_type == "uppercase":
            return actual_text.upper()
        elif format_type == "lowercase":
            return actual_text.lower()
        elif format_type == "titlecase":
            return actual_text.title()
        else:
            return "Invalid format type. Use uppercase, lowercase, or titlecase."
    except Exception as e:
        return f"Error formatting text: {str(e)}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for solving math expressions like '2 + 2', '10 * 5', '15 / 3'."
    ),
    Tool(
        name="Text Formatter",
        func=format_text,
        description="Useful for formatting text. Input format should be 'uppercase: text', 'lowercase: text', or 'titlecase: text'."
    )
]

prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:

{tools}

The available tool names are: {tool_names}

To use a tool, follow this format exactly:
Thought: Do I need to use a tool? Yes
Action: tool_name
Action Input: the input to the tool

After the tool responds, you will get:
Observation: tool result

You can repeat the Thought/Action/Observation steps if needed.

When you know the final answer, respond in this format:
Thought: I now know the final answer
Final Answer: the final answer to the user's question

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
test_questions = [
    "What is 25 + 63?", 
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7", 
    "titlecase: langchain is awesome",
]
for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"input": question})
    print("Final Output:", result["output"])