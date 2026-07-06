import os
import re
import logging

import requests
import trafilatura

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from readability import Document
from requests.adapters import HTTPAdapter
from tavily import TavilyClient
from urllib3.util.retry import Retry

# ==========================================================
# LOAD ENV
# ==========================================================

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# REQUEST SESSION WITH RETRY
# ==========================================================

session = requests.Session()

retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("http://", adapter)
session.mount("https://", adapter)

# ==========================================================
# WEB SEARCH TOOL
# ==========================================================


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily.

    Args:
        query: Search query.

    Returns:
        Top search results including title, URL and summary.
    """

    logger.info(f"Searching web: {query}")

    try:

        results = tavily.search(
            query=query,
            max_results=5
        )

        if not results.get("results"):
            return "No search results found."

        output = []

        for item in results["results"]:

            output.append(
                f"""
Title:
{item.get('title', 'No title')}

URL:
{item.get('url', 'No URL')}

Summary:
{item.get('content', 'No summary')[:350]}
"""
            )

        return "\n" + "-" * 80 + "\n".join(output)

    except Exception as e:

        logger.exception("Web search failed.")

        return f"Search failed.\nError: {str(e)}"


# ==========================================================
# SCRAPER TOOL
# ==========================================================


@tool
def scrape_url(url: str) -> str:
    """
    Scrape readable content from a webpage.

    Uses multiple extraction strategies:

    1. Trafilatura
    2. Readability
    3. BeautifulSoup fallback
    """

    logger.info(f"Scraping URL: {url}")

    headers = {
        "User-Agent":
        (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:

        response = session.get(
            url,
            headers=headers,
            timeout=20,
            verify=True
        )

        response.raise_for_status()

        html = response.text

        # ==================================================
        # Strategy 1 : Trafilatura
        # ==================================================

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:

            cleaned = re.sub(r"\s+", " ", extracted)

            logger.info("Trafilatura extraction successful.")

            return f"""
Source:
{url}

Content:
{cleaned[:7000]}
"""

        # ==================================================
        # Strategy 2 : Readability
        # ==================================================

        document = Document(html)

        clean_html = document.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form"
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if len(text) > 200:

            cleaned = re.sub(r"\s+", " ", text)

            logger.info("Readability extraction successful.")

            return f"""
Source:
{url}

Content:
{cleaned[:7000]}
"""

        # ==================================================
        # Strategy 3 : BeautifulSoup fallback
        # ==================================================

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form"
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        cleaned = re.sub(r"\s+", " ", text)

        if cleaned:

            logger.info("Fallback extraction successful.")

            return f"""
Source:
{url}

Content:
{cleaned[:7000]}
"""

        return "Could not extract meaningful content."

    except requests.exceptions.Timeout:

        logger.error("Timeout while scraping.")

        return "Request timed out while scraping the webpage."

    except requests.exceptions.ConnectionError:

        logger.error("Connection error.")

        return "Failed to connect to the website."

    except requests.exceptions.HTTPError as e:

        logger.error(str(e))

        return f"HTTP Error: {str(e)}"

    except requests.exceptions.RequestException as e:

        logger.error(str(e))

        return f"Request failed: {str(e)}"

    except Exception as e:

        logger.exception("Unexpected scraping error.")

        return f"Unexpected error while scraping.\nError: {str(e)}"