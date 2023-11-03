import streamlit as st
from predibase import PredibaseClient
from util import simulate_stream, chat_agents, PROFESSIONAL, MILITARY, FRIENDLY, DEFAULT

import logging

logger = logging.getLogger(__name__)

pc = PredibaseClient()

#########################
# Application Structure #
#########################

# Page title
st.title("Enhanced Chat System")

# Setup columns
col1, col2 = st.columns(2)

# Setup session state

if "intent" not in st.session_state:
    st.session_state["intent"] = None

###################
# Agent Selection #
###################
with col1:
    st.subheader("Tone Selection")
    agent_tone = st.radio(
        label="",
        # options=[DEFAULT, PROFESSIONAL, MILITARY, FRIENDLY],
        options=[DEFAULT],
    )
    current_agent = chat_agents[agent_tone]

#########################
# Intent Classification #
#########################
with col2:
    st.subheader("Intent")
    st.metric(label="", value=st.session_state.intent)

# Product Recommendations
st.subheader("Agent Chat")

print("CURRENT_AGENT: ", current_agent.external_chat_history)

# Display chat messages from history on app rerun
for message in current_agent.external_chat_history:
    print("TESTING")
    logger.info(current_agent.external_chat_history)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Say something"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    simulate_stream(current_agent.chat_completion(prompt))

    st.session_state.intent = current_agent.intent_classification(prompt)
    st.experimental_rerun()
