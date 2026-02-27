from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOll


def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case."""
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
            return "Invalid format type."
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
        description="Use format: 'uppercase: text', 'lowercase: text', or 'titlecase: text'."
    )
]

prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.

You have access to these tools:
{tools}

Tool names: {tool_names}

Follow this format:
Thought: Do I need a tool? Yes/No
Action: tool_name
Action Input: input
Observation: result

When finished:
Final Answer: answer

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Ollama LLM (Local Model)
llm = ChatOllama(
    model="llama3",
    temperature=0
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