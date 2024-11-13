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
import pandas as pd
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore(".embeddings")
base = OpenAIEmbeddings(model="text-embedding-3-small")
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
    """Embed the documents and persist the Chroma database."""
    docs = pd.read_excel("espr-rejections.xlsx")
    docs["Decision"] = docs["Final Decision Family"].replace({"Reject and Transfer": "Reject"})
    metadata = [
        {
            "key": d["Manuscript Number"],
            "h1": d["Article Title"],
            "decision": d["Decision"],
        }
        for _, d in docs.iterrows()
    ]
    documents = text_splitter.create_documents(
        [
            "\n\n".join((d["Article Title"], d["Abstract"], d["Keyword"]))
            for _, d in docs.iterrows()
        ],
        metadata,
    )
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="paperrejections",
    )


if __name__ == "__main__":
    embed()
