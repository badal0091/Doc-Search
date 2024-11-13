import httpx
import os
import pandas as pd
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm import tqdm
import yaml

# Cache OpenAI embeddings at .embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define the YAML file path in the parent directory
file_path = os.path.join(parent_dir, '.secrets.yaml')

# Load secrets from the YAML file
with open(file_path, 'r') as file:
    secrets = yaml.safe_load(file)

# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore(".embeddings")
base = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://gramener.com/llmproxy/v1/",
    openai_api_key=f'{secrets["DOCSEARCH_LLMPROXY_JWT"]}:docsearch',
)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)

# Chunk text into a reasonable number of tokens with some overlap
tokenizer = tiktoken.get_encoding("cl100k_base")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)


def embed():
    """Embed the documents and persist the Chroma database."""
    docs, metadata = [], []
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "wef.csv"))
    rows = list(data[data["Sector"] == "Finance"].iterrows())
    for _, row in tqdm(rows):
        target_file = row["URL"]
        if not os.path.exists(target_file):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
            response = httpx.get("https://www3.weforum.org/docs/" + row["URL"], headers=headers)
            with open(target_file, "wb") as f:
                f.write(response.content)
        loader = PyMuPDFLoader(target_file)
        pdf_pages = loader.load()
        for page in pdf_pages:
            m = page.metadata
            key = os.path.basename(row["Title"]) + f' p{m["page"] + 1}'
            docs.append(page)
            metadata.append({"key": row["URL"], "h1": key})
    documents = text_splitter.create_documents([doc.page_content for doc in docs], metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="wef",
    )


if __name__ == "__main__":
    embed()
