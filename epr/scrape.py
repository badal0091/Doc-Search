# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "requests",
#     "tiktoken",
#     "tqdm",
# ]
# ///
import os
import re
import requests
import urllib3
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm import tqdm
import tiktoken
import yaml

# Suppress the InsecureRequestWarning from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get the current script directory and parent directory
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
    chunk_size=300,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)


def sanitize_filename(filename):
    """
    Remove or replace invalid characters from the filename.

    Args:
        filename (str): The original filename.

    Returns:
        str: The sanitized filename with invalid characters replaced by underscores.
    """
    return re.sub(r'[<>:"/\\|?*%]', '_', filename)


def download_pdf(url):
    """
    Download a PDF from a given URL and save it locally.

    Args:
        url (str): The URL of the PDF file.

    Returns:
        str: The name of the saved PDF file.
    """
    # Disable SSL verification
    response = requests.get(url, verify=False, timeout=10)  # noqa: S501 certs don't work
    response.raise_for_status()  # Ensure the request was successful
    file_name = os.path.basename(url)
    file_name = sanitize_filename(file_name)  # Sanitize the filename
    with open(file_name, 'wb') as f:
        f.write(response.content)
    return file_name


def embed(pdf_urls):
    """
    Embed the documents from a list of PDF URLs and persist the Chroma database.

    Args:
        pdf_urls (list): A list of URLs pointing to PDF documents.

    Returns:
        Chroma: The Chroma vector store containing the embedded documents.
    """
    docs, metadata = [], []
    for row in tqdm(pdf_urls):
        local_pdf_path = download_pdf(row["url"])  # Download PDF first
        loader = PyMuPDFLoader(local_pdf_path)  # Load the local file with PyMuPDFLoader
        pdf_pages = loader.load()
        for page in pdf_pages:
            m = page.metadata
            key = row["url"] + f'#p{m["page"] + 1}'
            docs.append(page)
            metadata.append({"key": key, "h1": f"{row['bill']} p{m['page'] + 1}"})
        os.remove(local_pdf_path)  # Optionally delete the file after processing
    documents = text_splitter.create_documents([doc.page_content for doc in docs], metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="epr",
    )


if __name__ == "__main__":
    pdf_urls = [
        {
            "bill": "Illinois - State Bill 1555",
            "url": "https://www.ilga.gov/legislation/publicacts/103/PDF/103-0383.pdf",
        },
        {
            "bill": "Connecticut Senate Bill 311 (2023)",
            "url": "https://www.cga.ct.gov/2023/TOB/S/PDF/2023SB-00311-R00-SB.PDF",
        },
        {
            "bill": "Vermont House Bill 142 (2021)",
            "url": "https://legislature.vermont.gov/Documents/2022/Docs/JOURNAL/hj210127.pdf#page=2",
        },
        {
            "bill": "New Jersey Assembly Bill 1444 (2022)",
            "url": "https://pub.njleg.state.nj.us/Bills/2022/A1500/1444_I1.PDF",
        },
        {
            "bill": "Colorado House Bill 22-1355 Revised (2022)",
            "url": "https://leg.colorado.gov/sites/default/files/2022a_1355_signed.pdf",
        },
        {
            "bill": "Washington State Senate Bill 5022-S2.E (2021)",
            "url": "https://lawfilesext.leg.wa.gov/biennium/2021-22/Pdf/Bill%20Reports/Senate/5022%20SBR%20WM%20OC%2021.pdf?q=20240823015554",
        },
        {
            "bill": "Chapter 465 Maryland Senate Bill 222 (2023)",
            "url": "https://mgaleg.maryland.gov/2023RS/fnotes/bil_0002/sb0222.pdf",
        },
    ]

    embed(pdf_urls)
