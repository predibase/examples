import pandas as pd
import streamlit as st
from predibase import PredibaseClient

pc = PredibaseClient(
    token="INSERT TOKEN HERE"
)


def prod_to_str(prod: pd.Series):
    """
    This function re-formats product details from a tabular format to a string format that can be consumed by an
    LLM for generation tasks.

    :param prod: A pandas Series containing product details
    :return: A string containing product details
    """
    product_cols = {
        "prod_name",
        "product_type_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    }

    output_text = "\n"
    for key, val in prod.items():
        if key in product_cols:
            output_text += f"{key}: {val}, "

    return output_text


@st.cache_data
def generate_subject_line(
    prod: pd.Series,
    tone: str = "concise",
    llm: str = "vicuna-13b",
):
    """
    This function calls a Predibase hosted LLM and generates a subject line based on the specific product details
    provided.

    :param prod: A pandas Series containing product details.
    :param tone: The tone of the subject line to generate.
    :param llm: The name of the LLM to use for generation.
    :return: The generated subject line from the LLM.
    """
    prompt = f"""
        Generate a {tone.lower()} subject line for a personalized email outreach campaign advertising the following 
        product. Reply with the subject line only, no further explanation.
        Product information: '{prod_to_str(prod)}'.
        Subject:
    """
    resp = pc.prompt(prompt, llm, options={"max_new_tokens": 64}).response[0]
    return resp.lstrip("Subject: ")


@st.cache_data
def generate_greeting(
    prod: pd.Series,
    tone: str = "concise",
    llm: str = "vicuna-13b",
    context: str = "",
    customer_name: str = "",
):
    """
    This function calls a Predibase hosted LLM and generates a greeting based on the specific product details.

    :param prod: A pandas Series containing product details.
    :param tone: The tone of the subject line to generate.
    :param llm: The name of the LLM to use for generation.
    :param context: The previously generated sections of the email, provided so the LLM can generate a contiguous email.
    :param customer_name: Name of the customer to whom the email is being sent.
    :return: The generated greeting from the LLM.
    """
    prompt = f"""
        Generate a {tone.lower()} one sentence greeting for a personalized email outreach campaign. Reply with the 
        greeting only, do not write the closing of the email. 
        The email is being sent to {customer_name}.
        The product is a personalized recommendation.
        The subject line is: {context}.
        Do not repeat the previous context.
        Product information: '{prod_to_str(prod)}'.
        Do not repeat the instruction.
        One sentence greeting:
    """

    return pc.prompt(prompt, llm, options={"max_new_tokens": 128}).response[0]


@st.cache_data
def generate_prod_desc(
    prod: pd.Series,
    tone: str = "concise",
    llm: str = "vicuna-13b",
    context: str = "",
):
    """
    This function calls a Predibase hosted LLM and generates a product description based on the specific product
    details.

    :param prod: A pandas Series containing product details.
    :param tone: The tone of the subject line to generate.
    :param llm: The name of the LLM to use for generation.
    :param context: The previously generated sections of the email, provided so the LLM can generate a contiguous email.
    :return: The generated product description from the LLM.
    """
    prompt = f"""
        Product information: '{prod_to_str(prod)}'.
        The product description is part of an email and follows the following subject line and greeting: 
        ###
        {context}
        ###
        Do not repeat the previous subject line and greeting.
        Do not repeat the instruction.
        Write a {tone.lower()} product description for an advertisement of a product in a few sentences.
        Description:
    """

    return pc.prompt(prompt, llm, options={"max_new_tokens": 512}).response[0]


@st.cache_data
def generate_cta(
    tone: str = "concise",
    llm: str = "vicuna-13b",
    context: str = "",
):
    """
    This function calls a Predibase hosted LLM and generates a call to action based on the specific product details.

    :param tone: The tone of the subject line to generate.
    :param llm: The name of the LLM to use for generation.
    :param context: The previously generated sections of the email, provided so the LLM can generate a contiguous email.
    :return: The generated call to action from the LLM.
    """
    prompt = f"""
        The call to action follows the following context:
        ###
        {context}
        ###
        Do not repeat the previous context.
        Do not repeat the instruction.
        Write a {tone.lower()} one line call to action for the end of a personalized email advertisement of a product.
        One line call to action:
    """

    return pc.prompt(prompt, llm, options={"max_new_tokens": 512}).response[0]
