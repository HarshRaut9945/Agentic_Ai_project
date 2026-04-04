import streamlit as st
import re
from agent import get_agent_response

st.set_page_config(page_title="Chat Agent", layout='wide')
st.title("Advance Chat Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi — I'm your LangChain agent. Ask me anything!"}
    ]

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# User input
if user_input := st.chat_input("Type Your message"):
    # User message
    st.session_state.messages.append({"role": "user", "text": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response
    with st.spinner("Thinking..."):
        try:
            response = get_agent_response(user_input)
        except Exception as e:
            response = f"Agent error: {e}"

    st.session_state.messages.append({"role": "assistant", "text": response})

    with st.chat_message("assistant"):
        text = response

        yt_links = re.findall(r"(https?://www\.youtube\.com/watch\?v=[\w-]+)", text)

        for link in yt_links:
            col, _ = st.columns([2, 3])
            with col:
                st.video(link)
            text = text.replace(link, "")

        if text.strip():
            st.markdown(text)
