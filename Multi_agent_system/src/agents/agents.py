import os
import logging

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.tools.tools import web_search, scrape_url

# ==========================================================
# LOAD ENV
# ==========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ==========================================================
# SEARCH AGENT
# ==========================================================

search_system_prompt = """
You are a Search Agent.

Your ONLY responsibility is to search the web.

Guidelines:
- Use the web_search tool.
- Find reliable and recent information.
- Return useful search results.
- Include URLs whenever available.
- Do not invent facts.
"""

def build_search_agent():

    logger.info("Creating Search Agent...")

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=search_system_prompt,
    )

# ==========================================================
# READER AGENT
# ==========================================================

reader_system_prompt = """
You are a Reader Agent.

Your ONLY responsibility is to read webpages.

Guidelines:
- Use scrape_url tool.
- Extract only useful content.
- Ignore advertisements.
- Ignore menus.
- Ignore navigation bars.
- Ignore unnecessary page elements.
- Return clean readable text.
"""

def build_reader_agent():

    logger.info("Creating Reader Agent...")

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=reader_system_prompt,
    )

# ==========================================================
# WRITER CHAIN
# ==========================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research writer.

Write professional, structured and factual reports.

Rules:

- Only use the provided research.
- Never make up facts.
- Use professional language.
- Explain concepts clearly.
- Mention if information is missing.
""",
        ),
        (
            "human",
            """
Write a detailed research report.

Topic:
{topic}

Research:
{research}

Report Structure:

# Introduction

# Key Findings
(At least 3 detailed findings)

# Conclusion

# Sources
(List every source URL mentioned in the research.)
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()

# ==========================================================
# CRITIC CHAIN
# ==========================================================

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research critic.

Evaluate reports honestly.

Judge using:

- Accuracy
- Completeness
- Clarity
- Structure
- Source Quality

Provide constructive feedback.
""",
        ),
        (
            "human",
            """
Review this report.

Report:

{report}

Respond EXACTLY in this format.

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

Verdict:
...
""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()

# ==========================================================
# BUILD AGENTS
# ==========================================================

search_agent = build_search_agent()
reader_agent = build_reader_agent()

logger.info("All agents initialized successfully.")