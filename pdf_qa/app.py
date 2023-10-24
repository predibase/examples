import random
import time
from typing import Optional, List
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import HypotheticalDocumentEmbedder, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredPDFLoader
from html_templates import bot_template, css

st.session_state.settings = {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-ada-002",
    "large_language_model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "use_hyde": True,
    "hyde_llm": "gpt-3.5-turbo",
    "hyde_llm_temperature": 0.2,
    "generate_summary": True,
}


def get_pdf_text(pdf_docs) -> List[Document]:
    documents = []
    for pdf_doc in pdf_docs:
        # Write the doc to disk.
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, pdf_doc.name), "wb") as f:
                f.write(pdf_doc.read())

            # data = loader.load()
            loader = UnstructuredPDFLoader(os.path.join(temp_dir, pdf_doc.name))

            data = loader.load()
            documents.extend(data)
    return documents


def get_document_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Returns the list of text chunks from the given documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.create_documents(
        [document.page_content for document in documents],
        metadatas=[document.metadata for document in documents],
    )


def get_llm(llm_name: str, temperature: float):
    """Returns an LLM object corresponding to the provided LLM name, from advanced settings."""
    if llm_name == "gpt-3.5-turbo":
        return ChatOpenAI(temperature=temperature)
    if llm_name == "gpt-3.5-turbo-16k":
        return ChatOpenAI(temperature=temperature)
    if llm_name == "text-davinci-003":
        return OpenAI(model_name="text-davinci-003", temperature=temperature)
    return ChatOpenAI(temperature=temperature)


def get_vectorstore(
    chunks: List[Document],
    embedding_provider: str,
    embedding_model: str,
    use_hyde: bool,
    hyde_llm: Optional[str] = None,
    hyde_llm_temperature: Optional[float] = None,
):
    """Returns a FAISS vector store for the given text chunks."""
    if embedding_provider == "openai":
        base_embeddings = OpenAIEmbeddings(model=embedding_model)
    else:
        base_embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model)

    if use_hyde:
        embeddings = HypotheticalDocumentEmbedder.from_llm(
            get_llm(hyde_llm, hyde_llm_temperature), base_embeddings, "web_search"
        )
        return FAISS.from_texts(
            texts=[chunk.page_content for chunk in chunks],
            embedding=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )
    return FAISS.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        embedding=base_embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )


def get_retrieval_qa_chain(vectorstore, llm_name: str, temperature: float):
    """Returns the retrieval QA chain."""
    llm = get_llm(llm_name, temperature)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return qa


def handle_user_input(user_question: str, llm_object):
    """Issues a query to the vectorstore and displays the response."""
    response = llm_object({"query": user_question})

    st.write(
        bot_template.replace("{{MSG}}", response["result"]), unsafe_allow_html=True
    )
    with st.expander("Source attributions"):
        write_chunks(response["source_documents"])


def get_advanced_settings():
    """Returns advanced settings."""
    st.subheader("Indexing")

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=5000,
        value=1000,
        step=50,
        key="chunk_size",
        help="Chunking is the process of breaking down large pieces of text into smaller segments.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=10,
        key="chunk_overlap",
        help=(
            "Keep some overlap between chunks to make sure that the semantic context doesn’t get lost between "
            "chunks."
        ),
    )

    embedding_provider = st.radio(
        "Embedding provider",
        options=["openai", "huggingface"],
        key="embedding_provider",
        help="Choose which provider should be used for embedding document chunks.",
    )

    embedding_model_options = []
    if embedding_provider == "huggingface":
        embedding_model_options = [
            "intfloat/e5-large-v2",
            "hkunlp/instructor-xl",
            "hkunlp/instructor-large",
            "intfloat/e5-base-v2",
        ]
    else:
        embedding_model_options = ["text-embedding-ada-002"]
    embedding_model = st.selectbox(
        label="Embedding model",
        options=embedding_model_options,
        help=(
            "Choose which model is used for embedding. Refer to "
            "https://huggingface.co/spaces/mteb/leaderboard for a list of options. text-embedding-ada-002 is the best "
            "embedding model for OpenAI."
        ),
    )

    st.subheader("Querying")

    large_language_model = st.selectbox(
        "Large Language Model (LLM)",
        options=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "text-davinci-003",
            "vicuna-13b",
            "flan-t5-xxl",
            "redpajama-7b",
            "falcon-7b",
        ],
        key="llm",
        help="Choose which LLM is used for embedding, indexing, and querying.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=float(0.0),
        max_value=float(1.0),
        value=0.7,
        step=0.1,
        key="temperature",
        help="Temperature controls the randomness of language model output. Higher temperature results in more create "
        "outputs.",
    )

    use_hyde = st.checkbox(
        "Use HyDE",
        value=True,
        key="use_hyde",
        help="Whether to select the most relevant chunks according to Hypothetical Document Embeddings (HyDE). "
        "Read more about HyDE at https://python.langchain.com/docs/modules/chains/additional/hyde.",
    )

    if use_hyde:
        hyde_llm = st.selectbox(
            "HyDE Large Language Model (LLM)",
            options=[
                "gpt-3.5-turbo",
                "text-davinci-003",
            ],
            key="hyde_llm",
            help="The LLM used to generate hypothetical document reference documents to compare embeddings with.",
        )

        hyde_llm_temperature = st.slider(
            "HyDE LLM temperature",
            min_value=float(0.0),
            max_value=float(1.0),
            value=0.2,
            step=0.1,
            key="hyde_llm_temperature",
            help="Temperature for the LLM output for HyDE.",
        )
    else:
        hyde_llm = None
        hyde_llm_temperature = None

    generate_summary = st.checkbox(
        "Summarize",
        value=True,
        key="generate_summary",
        help="Whether to generate a summary of documents upon upload.",
    )

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "large_language_model": large_language_model,
        "temperature": temperature,
        "use_hyde": use_hyde,
        "hyde_llm": hyde_llm,
        "hyde_llm_temperature": hyde_llm_temperature,
        "generate_summary": generate_summary,
    }


def generate_summary(documents: List[Document]) -> str:
    """Returns a summary of the given document chunks."""
    with st.expander("Summary"):
        # Settings from
        # https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb
        with st.spinner("Summarizing (takes about 2 minutes)"):
            start_time = time.time()

            # Use big chunks to speed up summarization.
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=200
            )
            document_chunks = text_splitter.create_documents(
                [document.page_content for document in documents]
            )

            # Instantiate the summary chain.
            summary_chain = load_summarize_chain(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
                chain_type="map_reduce",
                verbose=True,
            )
            summary = summary_chain.run(document_chunks)
            end_time = time.time()
            st.write(summary)
            st.success(f"Summarized in {(end_time - start_time):.1f} seconds.")
            return summary


def get_sanitized_source(source_name: str) -> str:
    """Source names have tempdirs prepended to them. This function removes the tempdir."""
    return os.path.basename(source_name)


def write_chunks(document_chunks: List[Document]) -> None:
    """Streamlit writes the chunks as a table."""
    chunks_table = pd.DataFrame(
        {
            "source": [
                get_sanitized_source(document_chunk.metadata["source"])
                for document_chunk in document_chunks
            ],
            "text": [document_chunk.page_content for document_chunk in document_chunks],
        }
    )
    st.write(chunks_table)


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Predidoc: Ask your PDFs a question", page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    if "llm_object" not in st.session_state:
        st.session_state.llm_object = None
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "settings" not in st.session_state:
        st.session_state.settings = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    st.markdown(
        """
            <h1 style="text-align:center;">Predidoc</h1>
            <p style="text-align:center;">
            <img src="https://app.predibase.com/logos/predibase/predibase.svg" width="25" />
            Powered by <a href="https://predibase.com">Predibase</a>
            </p>
            """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf", "txt"],
        )

        if st.button("Process"):
            # Reset the summary.
            st.session_state.summary = None

            with st.spinner("Processing"):
                start_time = time.time()

                # Get pdf text.
                st.session_state.documents = get_pdf_text(docs)

                # Get the text chunks.
                st.session_state.document_chunks: List[Document] = get_document_chunks(
                    st.session_state.documents,
                    st.session_state.settings["chunk_size"],
                    st.session_state.settings["chunk_overlap"],
                )

                st.write(
                    f"Total number of chunks: {len(st.session_state.document_chunks)}"
                )

                end_time = time.time()

                st.success(f"Chunking took {(end_time - start_time):.1f} seconds.")

                # Create vector store and LLM object.
                start_time = time.time()
                vectorstore = get_vectorstore(
                    st.session_state.document_chunks,
                    st.session_state.settings["embedding_provider"],
                    st.session_state.settings["embedding_model"],
                    st.session_state.settings["use_hyde"],
                    st.session_state.settings["hyde_llm"],
                    st.session_state.settings["hyde_llm_temperature"],
                )
                st.session_state.llm_object = get_retrieval_qa_chain(
                    vectorstore,
                    st.session_state.settings["large_language_model"],
                    st.session_state.settings["temperature"],
                )

                end_time = time.time()

                st.success(
                    f"Vector store created in {(end_time - start_time):.1f} seconds."
                )
                print(
                    f"Vector store created with these settings: {st.session_state.settings}"
                )

                with st.expander("Sample chunks"):
                    sampled_chunks = random.sample(
                        st.session_state.document_chunks,
                        min(5, len(st.session_state.document_chunks)),
                    )
                    write_chunks(sampled_chunks)

                with st.expander("Document statistics"):
                    # Distribution of number of pages.
                    fig1, ax = plt.subplots()
                    # num_pages_array = np.array(list(doc_id_to_num_pages.values()))
                    ax.hist(
                        np.array(
                            [
                                len(document.page_content)
                                for document in st.session_state.document_chunks
                            ]
                        ),
                        bins=20,
                    )
                    st.write("Distribution of chunk sizes")
                    st.pyplot(fig1)

                    # Distribution of length of texts.
                    fig2, ax = plt.subplots()
                    ax.hist(
                        np.array(
                            [
                                len(document.page_content)
                                for document in st.session_state.documents
                            ]
                        ),
                        bins=20,
                    )
                    st.write("Distribution of document lengths (chars)")
                    st.pyplot(fig2)

        # Advanced settings.
        with st.expander("Advanced settings"):
            settings = get_advanced_settings()
            st.session_state.settings = settings

    # Summarization block.
    if (
        st.session_state.settings["generate_summary"]
        and st.session_state.llm_object
        and st.session_state.document_chunks
        and not st.session_state.summary
    ):
        st.session_state.summary = generate_summary(st.session_state.documents)
    elif st.session_state.summary:
        with st.expander("Summary"):
            st.write(st.session_state.summary)

    if not st.session_state.llm_object:
        st.warning("Connect your documents before asking a question.")

    # Handle user questions.
    user_question = st.text_input("Ask your documents a question.")
    if user_question and st.session_state.llm_object:
        with st.spinner("Querying"):
            start_time = time.time()
            handle_user_input(user_question, st.session_state.llm_object)
            end_time = time.time()
            st.success(f"Query took {(end_time - start_time):.1f} seconds.")


if __name__ == "__main__":
    main()
