import time
from io import BytesIO

import pandas as pd
import streamlit as st
from st_files_connection import FilesConnection

from util import (
    generate_cta,
    generate_greeting,
    generate_prod_desc,
    generate_subject_line,
)

# Preloaded recommendations for sample application
recs = pd.read_csv("recommendations.csv")
conn = st.experimental_connection("s3", type=FilesConnection)

# Sample customers to recommend products for
sample_customers = {
    "Samantha Monks": "1426b606af826fa9568b6fe75629d10bb6ff7ea40f93908c43113d991e512bd5",
    "Isabella Raines": "d74d7b234fa92d6f740bacb655655dcdd069d35272e412c05fbbbc4bfcba26bf",
    "Aurora Blackwood": "4de6988c4971cfc0c6910a85a54fa8d6fb11be64b87b98c46b73821d7db07ce7",
}


@st.cache_data
def present_product_recommendation(product_recommendation: pd.Series):
    """
    Function for displaying a product recommendation.

    :param product_recommendation: A pandas Series containing product info.
    :return: None
    """
    st.markdown(product_recommendation["prod_name"])

    with conn.open(product_recommendation["img_url"]) as img:
        st.image(BytesIO(img.read()), width=64)
    st.caption(product_recommendation["detail_desc"][:100] + "...")


def set_recommendation(index: int):
    """
    Function for setting the recommendation index in the session state.

    :param index: The index of the recommendation to set.
    :return: None
    """
    st.session_state["rec_idx"] = index


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


#########################
# Application Structure #
#########################

# Page title
st.title("Personalized Campaigns")

# User Selection
customer = st.radio(
    label="User",
    options=sample_customers.keys(),
    horizontal=True,
    key="customer",
)

# Product Recommendations
st.subheader("Recommend products")

customer_id = sample_customers[st.session_state.get("customer", "Samantha Monks")]
curr_recs = recs[recs.customer_id == customer_id]

for (index, recommendation), col in zip(
        curr_recs.drop_duplicates("article_id")[:3].iterrows(), st.columns(3)
):
    with col:
        present_product_recommendation(recommendation)
        st.button("Select", key=index, on_click=set_recommendation, args=[index])


# Email Generation Selection Options
st.subheader("Generate email")
col1, col2 = st.columns(2)
with col1:
    tone = st.radio("Tone", ("Concise", "Witty", "Friendly"), horizontal=True)
with col2:
    llm = st.radio(
        "Large Language Model", ("llama-2-13b", "vicuna-13b"), horizontal=True
    )

if "rec_idx" in st.session_state:
    rec = recs.loc[st.session_state["rec_idx"], ]

    # Email Subject Generation Section
    st.divider()
    st.caption("SUBJECT")
    subject = generate_subject_line(rec, tone, llm)
    simulate_stream(subject)
    st.divider()

    # Email Greeting Generation Section
    st.caption("GREETING")
    intro = generate_greeting(rec, tone, llm, subject, customer)
    simulate_stream(intro)

    # Email Body Generation Section
    st.caption("PRODUCT DESCRIPTION")
    desc = generate_prod_desc(rec, tone, llm, subject + intro)
    simulate_stream(desc)

    # Email CTA Generation Section
    st.caption("CTA")
    resp = generate_cta(rec, tone, llm, subject + intro + desc)
    simulate_stream(resp)
    st.divider()
