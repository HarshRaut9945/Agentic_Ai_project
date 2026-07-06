import time
import streamlit as st

from src.pipelines.pipeline import run_research_pipeline

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("🔬 Multi-Agent Research Assistant")
st.markdown(
    """
Research any topic using multiple AI agents.

### 🤖 Agents
- 🔎 Search Agent
- 📖 Reader Agent
- ✍️ Writer Chain
- 🧐 Critic Chain
"""
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("⚙️ About")

    st.markdown("""
This project demonstrates a complete Multi-Agent AI workflow.

### Pipeline

Search Agent

↓

Reader Agent

↓

Writer Chain

↓

Critic Chain

---

### Tech Stack

- LangChain
- OpenAI GPT-4o-mini
- Tavily API
- BeautifulSoup
- Trafilatura
- Readability
- Streamlit
""")

# ==========================================================
# USER INPUT
# ==========================================================

topic = st.text_input(
    "Research Topic",
    placeholder="Example: Artificial General Intelligence"
)

col1, col2 = st.columns(2)

run_btn = col1.button(
    "🚀 Run Research",
    use_container_width=True
)

clear_btn = col2.button(
    "🗑 Clear",
    use_container_width=True
)

# ==========================================================
# CLEAR BUTTON
# ==========================================================

if clear_btn:

    st.session_state.clear()

    st.rerun()

# ==========================================================
# RUN PIPELINE
# ==========================================================

if run_btn:

    if topic.strip() == "":

        st.warning("Please enter a research topic.")

    else:

        start_time = time.time()

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info("🔎 Search Agent is working...")
            progress.progress(20)

            with st.spinner("Running Multi-Agent Pipeline..."):

                results = run_research_pipeline(topic)

            progress.progress(100)

            status.success("✅ Research Completed")

            end_time = time.time()

            st.success(
                f"Completed in {end_time-start_time:.2f} seconds"
            )

            # ==================================================
            # REPORT
            # ==================================================

            st.divider()

            st.header("📄 Research Report")

            st.markdown(results["report"])

            # ==================================================
            # FEEDBACK
            # ==================================================

            st.divider()

            st.header("🧐 Critic Feedback")

            st.markdown(results["feedback"])

            # ==================================================
            # RAW SEARCH RESULTS
            # ==================================================

            with st.expander("🔎 Search Results"):

                st.text(results["search_results"])

            # ==================================================
            # SCRAPED CONTENT
            # ==================================================

            with st.expander("📖 Scraped Content"):

                st.text(results["scraped_content"])

            # ==================================================
            # DOWNLOAD REPORT
            # ==================================================

            st.download_button(
                label="⬇ Download Report",
                data=results["report"],
                file_name=f"{topic.replace(' ','_')}_Research_Report.md",
                mime="text/markdown"
            )

        except Exception as e:

            st.error(f"❌ Error: {str(e)}")