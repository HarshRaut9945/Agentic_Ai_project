from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import os

from tool import (
    multi_web_search,
    youtube_search,
    weather_search,
    news_search,
)

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Tools
tools = [
    Tool(
        name="WebSearch",
        func=multi_web_search,
        description="Search web, Wikipedia and Arxiv"
    ),
    Tool(
        name="YouTubeSearch",
        func=youtube_search,
        description="Search YouTube videos"
    ),
    Tool(
        name="WeatherSearch",
        func=lambda q: weather_search(q, OPENWEATHER_API_KEY),
        description="Get weather info"
    ),
    Tool(
        name="NewsSearch",
        func=lambda q: news_search(q, NEWS_API_KEY),
        description="Get latest news"
    ),
]

# ✅ FIXED PROMPT (IMPORTANT)
prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.

Conversation so far:
{chat_history}

You have access to the following tools:
{tools}

Use this format:

Question: {input}
Thought: think step-by-step
Action: one of [{tool_names}]
Action Input: input
Observation: result
Final Answer: answer

{agent_scratchpad}
""")

# Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
     memory=memory 
)

# Function
# def get_agent_response(query: str) -> str:
#     try:
#         response = agent_executor.invoke({"input": query})
#         return response.get("output", "No response")
#     except Exception as e:
#         return f"Agent error: {e}"


def get_agent_response(query: str) -> str:
    try:
        # ✅ Direct LLM (fast, 1 API call only)
        if any(word in query.lower() for word in ["code", "program", "recursion", "factorial", "explain"]):
            return llm.invoke(query).content

        # ✅ Use agent only when needed
        response = agent_executor.invoke({"input": query})
        return response.get("output", "No response")

    except Exception as e:
        return f"Agent error: {e}"