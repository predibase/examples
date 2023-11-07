import streamlit as st
from predibase import PredibaseClient
from util import simulate_stream, chat_agents, PROFESSIONAL, MILITARY, FRIENDLY, DEFAULT

import logging

logger = logging.getLogger(__name__)

pc = PredibaseClient()


###############
# Reset State #
###############

def reset_state():
    print("RESETTING STATE")
    st.session_state["intent"] = None
    st.session_state.current_agent.reset()


#####################
# Application Setup #
#####################

# Page title
st.title("Enhanced Chat System")

# Setup session state
if "intent" not in st.session_state:
    st.session_state["intent"] = None
    st.session_state["current_agent"] = chat_agents[DEFAULT]

##########################
# Side Bar Configuration #
##########################
with st.sidebar:
    st.subheader("Tone Selection")
    agent_tone = st.radio(
        label="",
        # options=[DEFAULT, PROFESSIONAL, MILITARY, FRIENDLY],
        options=[DEFAULT, MILITARY],
    )
    st.session_state.current_agent = chat_agents[agent_tone]

    # Add spacing
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.subheader("Intent")
    st.metric(label="", value=st.session_state.intent)

######################
# Chat Configuration #
######################
st.subheader("Agent Chat")

# Display chat messages from history on app rerun
for message in st.session_state.current_agent.external_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    simulate_stream(st.session_state.current_agent.chat_completion(prompt))
    print("CHAT HISTORY: \n", "\n".join(st.session_state.current_agent.internal_chat_history))
    st.session_state.intent = st.session_state.current_agent.intent_classification(prompt)
    st.experimental_rerun()
