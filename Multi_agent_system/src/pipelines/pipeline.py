from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:
    """
    Executes the complete multi-agent research workflow.

    Steps:
    1. Search Agent
    2. Reader Agent
    3. Writer Chain
    4. Critic Chain
    """

    state = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
    }

    try:

        # ==========================================================
        # STEP 1 : SEARCH AGENT
        # ==========================================================

        print("\n" + "=" * 70)
        print("STEP 1 : SEARCH AGENT")
        print("=" * 70)

        search_agent = build_search_agent()

        search_response = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"""
Find recent, reliable and detailed information about:

{topic}

Use the web_search tool.

Return titles, URLs and useful summaries.
""",
                    )
                ]
            }
        )

        state["search_results"] = (
            search_response["messages"][-1].content
            if search_response.get("messages")
            else ""
        )

        print(state["search_results"][:1000])

    except Exception as e:

        state["search_results"] = f"Search Agent Error: {e}"

        print(state["search_results"])

    try:

        # ==========================================================
        # STEP 2 : READER AGENT
        # ==========================================================

        print("\n" + "=" * 70)
        print("STEP 2 : READER AGENT")
        print("=" * 70)

        reader_agent = build_reader_agent()

        reader_response = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"""
Below are search results.

Select the BEST URL.

Scrape it using scrape_url.

Search Results:

{state["search_results"]}
""",
                    )
                ]
            }
        )

        state["scraped_content"] = (
            reader_response["messages"][-1].content
            if reader_response.get("messages")
            else ""
        )

        print(state["scraped_content"][:1000])

    except Exception as e:

        state["scraped_content"] = f"Reader Agent Error: {e}"

        print(state["scraped_content"])

    try:

        # ==========================================================
        # STEP 3 : WRITER
        # ==========================================================

        print("\n" + "=" * 70)
        print("STEP 3 : WRITER")
        print("=" * 70)

        research = f"""

SEARCH RESULTS

{state["search_results"]}

--------------------------------------------------------

SCRAPED CONTENT

{state["scraped_content"]}

"""

        state["report"] = writer_chain.invoke(
            {
                "topic": topic,
                "research": research,
            }
        )

        print(state["report"][:1500])

    except Exception as e:

        state["report"] = f"Writer Error: {e}"

        print(state["report"])

    try:

        # ==========================================================
        # STEP 4 : CRITIC
        # ==========================================================

        print("\n" + "=" * 70)
        print("STEP 4 : CRITIC")
        print("=" * 70)

        state["feedback"] = critic_chain.invoke(
            {
                "report": state["report"],
            }
        )

        print(state["feedback"])

    except Exception as e:

        state["feedback"] = f"Critic Error: {e}"

        print(state["feedback"])

    return state