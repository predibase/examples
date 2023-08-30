import os
import concurrent.futures

import streamlit as st
from time import perf_counter
from predibase import PredibaseClient
from info_extract import Corpus
from info_extract.endpoints import get_llm_endpoint
from info_extract.retrieval import get_retriever


def try_get_fields(dataset):
    try:
        fields = [field.name for field in dataset.get_fields()]
        if "document_id" in fields and "document_name" in fields and "document_text" in fields:
            return (dataset.name, dataset.connection.name)
    except Exception:
        return None


def list_rag_datasets(pc: PredibaseClient):
    dataset_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(connection.list_datasets) for connection in pc.list_connections()]
        for future in concurrent.futures.as_completed(futures):
            try:
                dataset_list.extend(future.result())
            except Exception as exc:
                print("ERROR:", exc)

    rag_dataset_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(try_get_fields, dataset) for dataset in dataset_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                rag_dataset_list.append(future.result())
            except Exception as exc:
                print("ERROR:", exc)
        rag_dataset_list = [elem for elem in rag_dataset_list if elem is not None]

    return rag_dataset_list


st.set_page_config(page_title="Predibase Document Question Answering", page_icon="📖", layout="wide")
st.header("📖 Predibase Document QA")


with st.sidebar:
    st.markdown("""
    ## How to use
    0. Enter your Predibase API token.
    1. Connect your dataset in Predibase.
    2. Select the dataset you'd like to query.
    3. Ask questions about your documents.
        """)

    api_key_input = st.text_input(
        "Predibase API Token",
        type="password",
        placeholder="Paste your Predibase API token here",
        help="You can get your API key from https://app.predibase.com/settings",  # noqa: E501
        value=os.environ.get("PREDIBASE_API_TOKEN", None) or st.session_state.get("PREDIBASE_API_TOKEN", ""),
    )
    st.session_state["PREDIBASE_API_TOKEN"] = api_key_input


predibase_api_token = st.session_state.get("PREDIBASE_API_TOKEN")
print("GOT THE PREDIBASE API TOKEN")


@st.cache_data(show_spinner=False)
def get_rag_datasets(predibase_api_token):
    print("CREATED THE PREDIBASE CLIENT")
    rag_dataset_list = list_rag_datasets(pc)
    print("GOT THE RAG DATASETS")
    rag_tup = tuple([elem[0] for elem in rag_dataset_list])
    return rag_tup, rag_dataset_list


@st.cache_data(show_spinner=False)
def build_corpus(dataset_name, connection_name):
    predibase_dataset = pc.get_dataset(dataset_name=dataset_name, connection_name=connection_name)
    print("GETTING DATASET", predibase_dataset.name)
    corpus_name = predibase_dataset.name
    chunk_size = 1999

    llm_endpoint = get_llm_endpoint(model_provider="predibase", model_name="llama-2-13b", predibase_client=pc)

    # Use Predibase infrastructure for indexing and retrieval
    retriever = get_retriever(retrieval_provider="predibase", index_name=f"{corpus_name}-{chunk_size}",
                              predibase_client=pc, model_name="llama-2-13b")

    # Create the corpus of documents and pass in the necessary resources (LLM and retriever)
    corpus = Corpus(predibase_dataset.to_dataframe(), name=corpus_name, llm_endpoint=llm_endpoint, retriever=retriever)
    chunks = corpus.chunk(chunk_size)

    with st.spinner("Indexing corpus... This may take a while"):
        print("INDEXING")
        corpus.index()
    return corpus


if not predibase_api_token:
    st.warning("Enter your Predibase API token in the sidebar. You can get a key at https://app.predibase.com/settings")
else:
    pc = PredibaseClient(token=predibase_api_token)
    rag_tup, rag_dataset_list = get_rag_datasets(predibase_api_token)
    selected_dataset_name = st.selectbox("Select the dataset you'd like to ask questions about", rag_tup)
    print("THE SELECTED CORPUS IS", selected_dataset_name)
    dataset_name, connection_name = None, None
    for elem in rag_dataset_list:
        if elem[0] == selected_dataset_name:
            dataset_name, connection_name = elem
            break

    corpus = build_corpus(dataset_name, connection_name)


with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the corpus")
    submit = st.form_submit_button("Submit")
    print("SUBMIT:", submit)


if submit:
    progress_text = "Retrieving, Extracting answer, Formulating a response"

    print("JUST BEFORE corpus.query")
    start_t = perf_counter()

    with st.spinner(text=progress_text):
        rag_response = corpus.query(query)

    print("GOT THE ANSWER", rag_response.answer)
    print("took", perf_counter() - start_t)

    st.markdown("### Answer")
    st.markdown(rag_response.answer)

    st.markdown("### Sources")
    for source in rag_response.chunk_answers:
        st.markdown(f"Document ID: {source.document_id}")
        st.markdown(f"Text: {source.chunk_text}")
        st.markdown("---")
