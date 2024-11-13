# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "pandas",
#     "tiktoken",
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
# os.environ["OPENAI_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImJhZGFsLnZhcnNobmV5QHN0cmFpdmUuY29tIn0.ysCmDtq_uD4OUQghTCobhldQFim9ufNG4vSGgb21UXI"
# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore(".embeddings")
base = OpenAIEmbeddings(model="text-embedding-3-small", api_key=f'{"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImJhZGFsLnZhcnNobmV5QHN0cmFpdmUuY29tIn0.ysCmDtq_uD4OUQghTCobhldQFim9ufNG4vSGgb21UXI"}', base_url="https://llmfoundry.straive.com/openai/v1/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)

# Chunk text into a reasonable number of tokens with some overlap
tokenizer = tiktoken.get_encoding("cl100k_base")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)

def embed():
    """Embed the resumes and persist the Chroma database."""
    docs = pd.read_csv("UpdatedResumeDataSet.csv")  # Replace with your CSV file path
    metadata = [
        {
            "category": d["Category"],
            "resume": d["Resume"]
        }
        for _, d in docs.iterrows()
    ]
    documents = text_splitter.create_documents(
        [d["Resume"] for _, d in docs.iterrows()],
        metadata,
    )
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="resumesearch",
    )

    
if __name__ == "__main__":
    embed()
