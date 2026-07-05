import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.agents import create_agent

# ==========================================================
# LOAD ENV VARIABLES
# ==========================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Assistant")
st.write("Search + Weather AI Agent using Gemini")

# ==========================================================
# SEARCH TOOL
# ==========================================================

search_tool = TavilySearchResults(max_results=2)

# ==========================================================
# WEATHER TOOL
# ==========================================================


@tool
def get_weather_data(city: str) -> str:
    """
    Returns current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather information for {city}"

    current = data["current"]

    return (
        f"City: {city}\n"
        f"Temperature: {current['temperature']} °C\n"
        f"Weather: {current['weather_descriptions'][0]}\n"
        f"Humidity: {current['humidity']}%\n"
        f"Wind Speed: {current['wind_speed']} km/h"
    )


# ==========================================================
# LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ==========================================================
# TOOLS
# ==========================================================

tools = [
    search_tool,
    get_weather_data,
]

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

system_prompt =  """
You are an AI assistant.

You have access to two tools.

1. Tavily Search
2. get_weather_data

IMPORTANT:

Whenever user asks about

- weather
- temperature
- humidity
- rain
- climate

You MUST call get_weather_data.

Never answer weather from memory.

Always use the tool.

After using the tool, summarize the tool output for the user.
"""

# ==========================================================
# CREATE AGENT
# ==========================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

# ==========================================================
# USER INPUT
# ==========================================================

user_query = st.text_input(
    "Ask anything...",
    placeholder="Example: Find the capital of India and its current weather",
)

# ==========================================================
# RUN
# ==========================================================

if st.button("Run Agent"):

    if not user_query.strip():
        st.warning("Please enter a query.")

    else:

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ]
                    }
                )

                answer = response["messages"][-1].content

                st.success("Done!")

                st.markdown("### Response")

                st.write(answer)

            except Exception as e:

                st.error(str(e))