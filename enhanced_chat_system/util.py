import time
import streamlit as st
from predibase import PredibaseClient
from chat_agent import ChatAgent

pc = PredibaseClient()

DEFAULT = "Default"
PROFESSIONAL = "Professional"
MILITARY = "Military"
FRIENDLY = "Friendly"


##########################
# Initialize Chat Agents #
##########################

chat_agents = {
    DEFAULT: ChatAgent(pc.LLM("pb://deployments/llama-2-13b-chat"), adapter=False),
    MILITARY: ChatAgent(pc.LLM("pb://deployments/llama-2-13b-chat"), adapter=True),
    # MILITARY: ChatAgent(pc.get_model("Tone Matching Models", version=1)),
    # PROFESSIONAL: ChatAgent(pc.get_model("Enhanced Chat System", version=2)),
    # FRIENDLY: ChatAgent(pc.get_model("Enhanced Chat System", version=3)),
}


def simulate_stream(generated_component: str):
    """
    Function for animating generation of text for the email components.

    :param generated_component: The generated text to animate.
    :return: None
    """
    # Simulate stream of response with milliseconds delay
    message_placeholder = st.empty()
    full_response = ""
    for chunk in generated_component.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
