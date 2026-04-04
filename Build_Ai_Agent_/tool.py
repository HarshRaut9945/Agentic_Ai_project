from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
    OpenWeatherMapAPIWrapper,
)

import requests


# -----------------------
# Web Search
# -----------------------


def multi_web_search(query: str) -> str:
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(backend='lite')
        ddg = DuckDuckGoSearchRun(api_wrapper=wrapper)
        wiki = WikipediaAPIWrapper()

        return "\n\n".join([
            "DuckDuckGo:\n" + ddg.run(query),
            "Wikipedia:\n" + wiki.run(query),
        ])

    except Exception as e:
        return f"Search error: {e}"


# -----------------------
# YouTube Search
# -----------------------
def youtube_search(query: str) -> str:
    try:
        yt = YouTubeSearchTool()
        return yt.run(query)
    except Exception as e:
        return f"YouTube error: {e}"


# -----------------------
# Weather Tool
# -----------------------
def weather_search(query: str, api_key: str) -> str:
    try:
        weather = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=api_key
        )
        return weather.run(query)
    except Exception as e:
        return f"Weather error: {e}"


# -----------------------
# News Tool
# -----------------------
def news_search(query: str, api_key: str) -> str:
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": 5,
        }

        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()

        data = res.json()
        articles = data.get("articles", [])

        return "\n".join(
            f"{a.get('title')} — {a.get('url')}"
            for a in articles
        ) or "No news found."

    except Exception as e:
        return f"News error: {e}"
    


    # Function
# def get_agent_response(query: str) -> str:
#     try:
#         response = agent_executor.invoke({"input": query})
#         return response.get("output", "No response")
#     except Exception as e:
#         return f"Agent error: {e}"