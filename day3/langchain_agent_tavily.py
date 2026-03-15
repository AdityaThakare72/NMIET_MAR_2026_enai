import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0  # Zero temperature for logical consistency in agents
)

# 3. Setup the Senses (Tavily Tool)
# 'max_results=3' keeps the context clean and token-efficient
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# 4. The Manual ReAct Prompt (The "Alternative to Hub")
# This defines the exact logic the agent uses to "think"
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final response to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

custom_prompt = PromptTemplate.from_template(template)

# 5. Construct the Agent
# This combines the LLM, Tools, and our custom manual prompt
agent = create_react_agent(llm, tools, custom_prompt)

# 6. Create the Executor (The Runtime Environment)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,               # Set to True to see the "Thought" process
    handle_parsing_errors=True, # Essential for handling minor LLM formatting slips
    max_iterations=5            # Safety limit to prevent infinite loops
)

# 7. Execute a Query
query = "What is the current weather in Pune and is it a good day for an outdoor trek?"
response = agent_executor.invoke({"input": query})

print(response["output"])