# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "tiktoken",
# ]
# ///
import os
import tiktoken
from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader
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
    chunk_size=300,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)


def embed():
    """Embed the documents and persist the Chroma database."""
    docs, metadata = [], []
    for path in glob("LM/**/*.pdf"):
        loader = PyMuPDFLoader(path)
        pdf_pages = loader.load()
        for page in pdf_pages:
            m = page.metadata
            key = os.path.basename(m["file_path"]) + f' p{m["page"] + 1}'
            docs.append(page)
            metadata.append({"key": key, "h1": key})
    documents = text_splitter.create_documents([doc.page_content for doc in docs], metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="learningmatepolicies",
    )


if __name__ == "__main__":
    embed()
