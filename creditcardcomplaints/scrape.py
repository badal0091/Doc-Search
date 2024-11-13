# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "pandas",
#     "tiktoken",
#     "tqdm",
# ]
# ///
import os
import pandas as pd
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from urllib.parse import urlencode
from urllib.request import urlretrieve

# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore('.embeddings')
base = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)

# Chunk text into a reasonable number of tokens with some overlap
tokenizer = tiktoken.get_encoding("cl100k_base")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)


def get_data():
    if not os.path.exists("data.csv"):
        params = {
            "consumer_consent_provided": "Consent provided",
            "date_received_max": "2024-03-27",
            "date_received_min": "2023-12-27",
            "field": "all",
            "format": "csv",
            "has_narrative": "true",
            "no_aggs": "true",
            "product": "Credit card",
            "size": "4301",
            "sort": "created_date_desc",
        }
        url = (
            "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?"
            + urlencode(params)
        )
        urlretrieve(url, "data.csv")  # noqa: S310 this is a safe URL


def embed():
    """Embed the documents and persist the Chroma database."""
    docs = pd.read_csv("data.csv").drop_duplicates(subset=["Consumer complaint narrative"])
    metadata = [
        {"key": d["Complaint ID"], "h1": f'{d["Company"]} {d["Sub-product"]} {d["Sub-issue"]}'}
        for _, d in docs.iterrows()
    ]
    documents = text_splitter.create_documents(
        [d["Consumer complaint narrative"] for _, d in docs.iterrows()], metadata
    )
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory='.chromadb',
        collection_name='creditcardcomplaints',
    )


if __name__ == "__main__":
    get_data()
    embed()
