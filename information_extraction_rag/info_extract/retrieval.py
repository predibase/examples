import os
from time import perf_counter
from typing import Optional

import pandas as pd
from ludwig.backend import initialize_backend
from ludwig.models.retrieval import SemanticRetrieval
from predibase import PredibaseClient

from info_extract.defaults import DEFAULT_CACHE_DIR


class Retriever:
    def __init__(self, **kwargs):
        pass

    def index(self, df_to_index: pd.DataFrame):
        pass

    def load_index(self):
        pass

    def retrieve(self, query: str, k: int):
        pass


class LudwigRetriever:
    def __init__(self, index_name: Optional[str] = None, cache_dir: Optional[str] = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.index_name = index_name
        self.semantic_retrieval = None

    def index(self, df_to_index: pd.DataFrame):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        print(f"Indexing {len(df_to_index)} chunks.")
        start_t = perf_counter()
        self.semantic_retrieval = SemanticRetrieval(model_name="all-mpnet-base-v2")
        backend = initialize_backend("local")
        self.semantic_retrieval.create_dataset_index(df_to_index, backend=backend, columns_to_index=["chunk_text"])
        end_t = perf_counter()
        print(f"\nTOOK {end_t - start_t}s to compute embeddings for the index.")
        print(f"Saving index to {self.cache_dir} under name {self.index_name}.")
        self.semantic_retrieval.save_index(name=self.index_name, cache_directory=self.cache_dir)

    def load_index(self):
        print(f"Loading index {self.index_name}.")
        start_t = perf_counter()
        self.semantic_retrieval = SemanticRetrieval(model_name="all-mpnet-base-v2")
        self.semantic_retrieval.load_index(name=self.index_name, cache_directory=self.cache_dir)
        end_t = perf_counter()
        print(f"\nTOOK {end_t - start_t}s to load the index.")

    def retrieve(self, query: str, k: int):
        if self.semantic_retrieval is None:
            try:
                self.load_index()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve the index `{self.index_name}` from `{self.cache_dir}`."
                    f" Please call `index` first."
                )

        backend = initialize_backend("local")
        answer = self.semantic_retrieval.search(
            df=pd.DataFrame({"query": [query]}), backend=backend, k=k, return_data=True
        )
        retrieved_documents = pd.DataFrame(answer[0])
        return retrieved_documents


class PredibaseRetriever:
    def __init__(
        self,
        predibase_client: PredibaseClient,
        index_name: Optional[str] = None,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        model_name: str = "llama-2-13b",
    ):
        self.cache_dir = cache_dir
        self.index_name = index_name
        self.predibase_client = predibase_client
        self.model_name = model_name

    def index(self, df_to_index: pd.DataFrame):
        try:
            index = self.predibase_client.get_dataset(self.index_name, connection_name="file_uploads")
        except Exception:
            self.predibase_client.create_dataset_from_df(df_to_index, name=self.index_name)
            index = self.predibase_client.get_dataset(self.index_name, connection_name="file_uploads")

        self.predibase_client.prompt("", self.model_name, index=index)

    def load_index(self):
        pass

    def retrieve(self, query: str, k: int):
        index = self.predibase_client.get_dataset(self.index_name, connection_name="file_uploads")
        return self.predibase_client.prompt(query, self.model_name, options={"retrieve_top_k": k}, index=index)


def get_retriever(retrieval_provider, **kwargs):
    if retrieval_provider == "predibase":
        return PredibaseRetriever(**kwargs)
    elif retrieval_provider == "ludwig":
        return LudwigRetriever(**kwargs)
