import concurrent.futures
import textwrap
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Any, List, Optional, Union

import pandas as pd

from info_extract.endpoints import LLMEndpoint
from info_extract.retrieval import Retriever
from info_extract.templates import (
    EXTRACT_TEMPLATE,
    FINAL_SYNTHESIZE_TEMPLATE,
    MULTIVERIFY_TEMPLATE,
    SYNTHESIZE_TEMPLATE,
)


@dataclass
class ChunkExtractionResult:
    """Dataclass to hold the extraction result for a chunk."""

    document_id: int
    chunk_id: int
    chunk_text: str
    query: str
    answer: str
    is_correct: bool


@dataclass
class RAGResult:
    """Dataclass to hold the result of a RAG (Retrieval-augmented generation) query."""

    answer: str
    chunk_answers: List[ChunkExtractionResult]


def chunk_text(text_input: str, overlap: bool = False, chunk_size: int = 2048):
    """Create chunks out of the provided text input.

    Args:
        text_input: input text to be chunked.
        overlap: whether there will be an overlap between the chunks.
        chunk_size: an upper bound on the number of characters per chunk.
    """
    chunks = textwrap.wrap(text_input, width=chunk_size)
    chunks_shifted = textwrap.wrap(text_input, width=chunk_size // 2)

    if overlap:
        chunks_shifted_concat = [chunks_shifted[0]]
        for i in range(1, len(chunks_shifted) - 1, 2):
            chunks_shifted_concat.append(chunks_shifted[i] + " " + chunks_shifted[i + 1])
        chunks_shifted_concat.append(chunks_shifted[-1])
        chunks += chunks_shifted_concat
    return chunks


def trimmer(seq: List[Any], size: int, filler: Any = "UNDEFINED"):
    """Pad list with filler up to a certain size.

    Args:
        seq: the list to be padded.
        size: the size of the list up to which to pad.
        filler: value to pad the list with.
    """
    return list(islice(chain(seq, repeat(filler)), size))


class Chunk:
    def __init__(self, document_id: int, chunk_id: int, chunk_text: str, llm_endpoint: LLMEndpoint):
        """Class to store chunk attributes."""
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
        self.llm_endpoint = llm_endpoint

    def extract(self, queries: List[str], do_llm_verify: bool = False) -> List[ChunkExtractionResult]:
        """Extract information from a chunk based on a number of queries.

        Args:
            queries: list of queries to use for extraction.
            do_llm_verify: verify whether the extracted answers are correct or not.
        Returns:
            List of ChunkExtractionResult. Each element corresponds to a query.
        """
        # extract LLM call
        answers = self.get_answer_given_chunk(queries)
        # (optional) verify LLM call
        verifications = self.verify_answer_given_query_and_chunk(queries, answers, do_llm_verify=do_llm_verify)

        chunk_extraction_result_list = []
        for q, a, v in zip(queries, answers, verifications):
            chunk_extraction_result = ChunkExtractionResult(
                document_id=self.document_id,
                chunk_id=self.chunk_id,
                chunk_text=self.chunk_text,
                query=q,
                answer=a,
                is_correct=v,
            )
            chunk_extraction_result_list.append(chunk_extraction_result)
        return chunk_extraction_result_list

    def get_answer_given_chunk(self, queries: List[str]) -> List[str]:
        """Extract information from a chunk based on a number of queries.

        Args:
            queries: list of queries to use for extraction.
        Returns:
            List of answers (strings). Each element corresponds to a query.
        """
        # create the formatted string containing all the questions.
        formatted_questions_list = [f"Q{i + 1}: {query}".strip() for i, query in enumerate(queries)]
        formatted_questions = "\n".join(formatted_questions_list)
        num_questions = len(formatted_questions_list)

        prompt = EXTRACT_TEMPLATE.format(self.chunk_text, formatted_questions)
        text = self.llm_endpoint.hit(prompt)
        text = [item.strip() for item in text.split("\n") if item.strip().startswith("A")]
        text = "A1: " + "\n".join(text)
        answers_list = text.split("\nA")
        answers_list = [ans[ans.find(":") + 1 :].strip().replace("\n", ". ") for ans in answers_list]
        num_answers = len(answers_list)

        if num_answers < num_questions:
            # todo replace with logging.warn
            print(
                f"In get_answer_given_chunk. num_answers: {num_answers} and num_questions: {num_questions} "
                f"for chunk {self.chunk_id} from document {self.document_id}."
            )
        answers_list = trimmer(answers_list, num_questions)

        return answers_list

    def verify_answer_given_query_and_chunk(self, queries: List[str], answers: List[str], do_llm_verify: bool = False):
        """Verify that the extracted information from a chunk based on a number of queries is correct and isn't a
        hallucination.

        Args:
            queries: list of queries to use for extraction.
            answers: list of answers to verify.
            do_llm_verify: whether to use an LLM to verify or just return True for the answer.
        Returns:
            List of booleans indicating whether the answer is True or False. Each element corresponds to a query.
        """

        assert len(queries) == len(answers)
        num_questions = len(queries)

        if not do_llm_verify:
            verifications_list = [True for _ in range(num_questions)]
            assert len(verifications_list) == num_questions
            return verifications_list

        formatted_question_answers_list = [
            f"Q{i + 1}: {q}\nA{i + 1}: {a}\n" for i, (q, a) in enumerate(zip(queries, answers))
        ]
        formatted_question_answers = "\n".join(formatted_question_answers_list)

        prompt = MULTIVERIFY_TEMPLATE.format(self.chunk_text, formatted_question_answers)
        text = self.llm_endpoint.hit(prompt)
        text = "A1 ASSESSMENT: " + text
        verifications_list = text.strip().split("\n")
        verifications_list = [ans[ans.find(":") + 1 :].strip() for ans in verifications_list]
        verifications_list = [False if "false" in ans.lower() else True for ans in verifications_list]
        num_verifications = len(verifications_list)

        if num_verifications < num_questions:
            # todo replace with logging.warn
            print(
                f"In verify_answer_given_query_and_chunk. num_verifications: {num_verifications} and "
                f"num_questions: {num_questions} for chunk {self.chunk_id} from document {self.document_id}."
            )

        # pad with false, indicating that the answer we got is not coming from the context.
        verifications_list = trimmer(verifications_list, num_questions, filler=False)
        verifications_list = [bool(item) for item in verifications_list]
        assert len(verifications_list) == num_questions

        return verifications_list


class ExtractionResult:
    def __init__(self, extraction_result_df: pd.DataFrame, chunk_list):
        self.extraction_result_df = extraction_result_df
        self.extractions = self.extraction_result_df.drop(columns=["chunk_ids"])
        self.chunk_list = chunk_list

    def get_attribution(self, document_id: int, query: str) -> List[Chunk]:
        """Return a list of chunks which the final generated answer came from.

        Args:
            document_id: document ID.
            query: query to get the attributions for.
        """
        if document_id not in self.extraction_result_df["document_id"].values:
            raise ValueError(
                f"document_id `{document_id}` is not part of the relevant chunks. Please select a "
                f"relevant document_id from the `extractions` table."
            )

        chunk_ids = self.extraction_result_df[
            (self.extraction_result_df["query"] == query) & (self.extraction_result_df["document_id"] == document_id)
        ]["chunk_ids"].item()

        relevant_chunks = []
        for chunk in self.chunk_list:
            if chunk.document_id == document_id and chunk.chunk_id in chunk_ids:
                relevant_chunks.append(chunk)
        return relevant_chunks


class ChunkList:
    def __init__(self, chunks_df: pd.DataFrame, llm_endpoint: LLMEndpoint, retriever: Retriever):
        """Initialization method for ChunkList, an interface for working with chunks."""
        self.df = chunks_df

        # Ludwig retriever.
        self.semantic_retrieval = None
        self.llm_endpoint = llm_endpoint
        self.retriever = retriever

    def chunk_list(self, df: Optional[pd.DataFrame] = None) -> List[Chunk]:
        """Return a list of Chunks."""
        if df is None:
            df = self.df

        for _, row in df.iterrows():
            yield Chunk(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                chunk_text=row["chunk_text"],
                llm_endpoint=self.llm_endpoint,
            )

    def extract(self, queries: List[str]):
        """Extract information from the chunks based on the queries.

        Args:
            queries: list of queries.
        """
        extraction_result_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(chunk.extract, queries) for chunk in self.chunk_list()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    extraction_result_list.extend(future.result())
                except Exception as exc:
                    raise exc

        return extraction_result_list

    def synthesize_extractions(self, query: str, answer_list: List[str]) -> Union[str, None]:
        """Use an LLM to synthesize an answer from a list of answers based on a query.

        Args:
            query: query to answer from chunks.
            answer_list: list of answers from each chunk to the given query.
        """
        valid_answers = ["- " + text for text in answer_list if "undefined" not in text.lower()]

        if len(valid_answers) == 0:
            return None

        formatted_answers = "\n".join(valid_answers)[:5000]
        prompt = SYNTHESIZE_TEMPLATE.format(formatted_answers, query)
        text = self.llm_endpoint.hit(prompt)

        return text

    def generate_per_document_extractions(self, extracted_df: pd.DataFrame) -> ExtractionResult:
        """Extracts per-document information from a dataframe with the following schema (document_id, chunk_id,
        chunk_text, query, answer, is_correct)

        Args:
            queries: list of queries to use to extract information.

        Returns:
            Pandas dataframe with the following columns (document_id, query, answer, chunk_ids)
        """
        document_id_list = list(set(extracted_df["document_id"].tolist()))
        extraction_result_list = []

        for document_id in document_id_list:
            filtered_df = extracted_df[extracted_df["document_id"] == document_id]
            query_list = list(set(filtered_df["query"].tolist()))
            for query in query_list:
                filtered_df = filtered_df[filtered_df["query"] == query]
                chunk_tuple_list = [
                    (chunk_id, answer)
                    for (chunk_id, answer) in zip(filtered_df["chunk_id"].tolist(), filtered_df["answer"].tolist())
                    if "undefined" not in answer.lower()
                ]
                chunk_id_list = [chunk_id for (chunk_id, _) in chunk_tuple_list]
                answer_list = [answer for (_, answer) in chunk_tuple_list]
                synthesized_text = self.synthesize_extractions(query, answer_list)

                if synthesized_text is not None:
                    entry = {
                        "document_id": document_id,
                        "query": query,
                        "answer": synthesized_text,
                        "chunk_ids": chunk_id_list,
                    }
                    extraction_result_list.append(entry)

        return ExtractionResult(
            extraction_result_df=pd.DataFrame(extraction_result_list), chunk_list=list(self.chunk_list())
        )

    def document_extract(self, queries: List[str]) -> ExtractionResult:
        """Extracts per-document information based on queries.

        Args:
            queries: list of queries to use to extract information.

        Returns:
            Pandas dataframe with the following columns (document_id, query, answer, chunk_ids)
        """
        extraction_result_list: List[ChunkExtractionResult] = self.extract(queries)
        self.most_recent_extracted_df: pd.DataFrame = pd.DataFrame(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_text": chunk.chunk_text,
                    "query": chunk.query,
                    "answer": chunk.answer,
                    "is_correct": chunk.is_correct,
                }
                for chunk in extraction_result_list
            ]
        )
        return self.generate_per_document_extractions(self.most_recent_extracted_df)

    def index(self):
        self.retriever.index(df_to_index=self.df)

    def load_index(self):
        self.retriever.load_index()

    def retrieve(self, query: str, topk: int) -> pd.DataFrame:
        """Retrieve topk chunks based on the query.

        Args:
            query: query to use for retrieval.
            topk: number of chunks to retrieve.

        Returns:
            Dataframe of retrieved documents.
        """
        return self.retriever.retrieve(query=query, k=topk)

    def query(self, query: str, topk: int = 10) -> RAGResult:
        """Retrieve, extract, and synthesize an anwer for a query from the chunks.

        Args:
            query: query to use for retrieval.
            topk: number of chunks to retrieve.

        Returns:
            Answer as a string and the relevant list of ChunkExtractionResult.
        """
        retrieved_documents = self.retrieve(query, topk)
        extraction_result_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(chunk.extract, [query]) for chunk in self.chunk_list(df=retrieved_documents)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    extraction_result_list.extend(future.result())
                except Exception as exc:
                    print("ERROR:", exc)

        return self.synthesize_rag(query, extraction_result_list)

    def synthesize_rag(self, query: str, extraction_result_list: List[ChunkExtractionResult]) -> RAGResult:
        """Synthesize an anwer for a query from the retrieved chunks.

        Args:
            query: query to use for retrieval.
            extraction_result_list: retrieved list of ChunkExtractionResult.

        Returns:
            Answer as a string and the relevant list of ChunkExtractionResult.
        """
        valid_answers = [
            "- " + chunk.answer
            for chunk in extraction_result_list
            if "undefined" not in chunk.answer.lower() and len(chunk.answer.strip()) > 0
        ]
        filtered_extraction_result_list = [
            chunk
            for chunk in extraction_result_list
            if "undefined" not in chunk.answer.lower() and len(chunk.answer.strip()) > 0
        ]
        if len(valid_answers) == 0:
            return RAGResult(
                answer=f"No answer found to the following query: {query}", chunk_answers=extraction_result_list
            )

        formatted_answers = "\n".join(valid_answers)[:5000]
        prompt = FINAL_SYNTHESIZE_TEMPLATE.format(formatted_answers, query)
        text = self.llm_endpoint.hit(prompt)

        return RAGResult(answer=text, chunk_answers=filtered_extraction_result_list)


class Corpus:
    def __init__(
        self,
        documents: Union[str, pd.DataFrame],
        name: str,
        llm_endpoint: LLMEndpoint,
        retriever: Optional[Retriever] = None,
    ):
        """Initialization method for the Corpus class, which holds documents and enables extraction and RAG.

        Args:
            documents: dataframe with the schema specified above, or base directory containing documents
                to be transformed (e.g. from PDF to text).
            name: name of the corpus.
            cache_dir: cache directory where the artifacts for the corpus (e.g. index) will be saved. Will override
                the default cache directory provided in the class constructor.
        """
        # documents is a df with columns "document_id", "document_name", "document_text".
        self.documents = self.create_documents_df(documents)
        # this will hold the chunks. Null initially.
        self.chunks: Union[ChunkList, None] = None
        self.name = name
        self.llm_endpoint = llm_endpoint
        self.retriever = retriever

    def create_documents_df(self, documents: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a dataframe storing the documents. The schema of the dataframe is the following: (document_id,
        document_name, document_text).

        Args:
             documents: dataframe with the schema specified above, or base directory containing documents
                to be transformed (e.g. from PDF to text).

        Returns:
            pd.DataFrame with the following schema: (document_id, document_name, document_text)
        """
        if isinstance(documents, pd.DataFrame):
            # todo: validate the the dataframe contains the right columns.
            return documents
        else:
            # todo: read PDFs and turn them into a df.
            return pd.DataFrame({})

    def chunk(self, chunk_size: int = 2048):
        """Create chunks out of the provided documents in the dataframe.

        Args:
            todo: make number of tokens.
            chunk_size: size of a chunk as number of characters.

        Returns:
            ChunkList object containing chunks.
        """
        document_chunks_df_list = []
        for _, row in self.documents.iterrows():
            document_id, document_name, document_text = row["document_id"], row["document_name"], row["document_text"]
            document_chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=False)
            num_chunks = len(document_chunks)
            document_chunks_df = pd.DataFrame(
                {
                    "chunk_id": list(range(num_chunks)),
                    "chunk_text": document_chunks,
                    "document_id": num_chunks * [document_id],
                }
            )
            document_chunks_df_list.append(document_chunks_df)
        self.chunks = ChunkList(
            chunks_df=pd.concat(document_chunks_df_list), llm_endpoint=self.llm_endpoint, retriever=self.retriever
        )
        self.chunk_size = chunk_size

        return self.chunks

    def index(self):
        """Create an embeddings index for the chunks created. must call `chunk` before.

        Args:
            cache_dir: cache directory where the index will be saved. Will override the default cache directory
                provided in the class constructor.
            index_name: index name under which the index will be saved. If not provided, it will default to
                f"{corpus name}-{chunk size}".
        """
        if self.retriever is None:
            raise RuntimeError("No retriever specified. Please pass a Retriever when constructing the `Corpus` object.")
        if self.chunks is None:
            raise RuntimeError("You must create chunks out of this corpus. Call the method `chunk` first.")

        self.chunks.index()

    def load_index(self):
        """Loads embedding index from cache directory.

        Args:
            cache_dir: cache directory where the index will be saved. Will override the default cache directory
                provided in the class constructor.
            index_name: index name under which the index will be saved. If not provided, it will default to
                f"{corpus name}-{chunk size}".
        """

        # todo: handle case where the chunks that the loaded index is for are not the same as the currently
        # stored chunks.

        if self.chunks is None:
            raise RuntimeError(
                "You must create chunks for this corpus before attempting to create an index. Call the method `chunk` first."
            )

        self.chunks.load_index()

    def extract(self, queries: List[str]) -> ExtractionResult:
        """Extract information from corpus based on the provided queries.

        Args:
            queries: list of queries to extract information for.
        """
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            raise TypeError("Argument `queries` must be a list of strings.")
        if self.chunks is None:
            raise RuntimeError(
                "You must create chunks for this corpus before attempting to perform extraction. Call the method `chunk` first."
            )

        return self.chunks.document_extract(queries=queries)

    def query(self, query: str, topk: int = 10) -> RAGResult:
        """Answer a query from the corpus. Uses a combination of retrieval and infomration extraction.

        Args:
            query: query to be answered from the corpus.
            topk: number of chunks to retrieve/get an answer from.

        Returns:
            string containing the answer.
        """
        result = self.chunks.query(query=query, topk=topk)
        return result
