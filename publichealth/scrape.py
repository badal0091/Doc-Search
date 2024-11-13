# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "langchain-chroma",
#     "langchain-community",
#     "pyMuPDF",
#     "tiktoken",
#     "tqdm",
# ]
# ///
import os
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm import tqdm
import tiktoken

# Get the current script directory and parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

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


def embed(pdf_urls):
    docs, metadata = [], []
    for row in tqdm(pdf_urls):
        loader = PyMuPDFLoader(row["url"])
        pdf_pages = loader.load()
        for page in pdf_pages:
            m = page.metadata
            key = row["url"] + f'#p{m["page"] + 1}'
            docs.append(page)
            metadata.append({"key": key, "h1": f"{row['title']} p{m['page'] + 1}"})
    documents = text_splitter.create_documents([doc.page_content for doc in docs], metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=os.path.join(
            current_dir, ".chromadb"
        ),  # Ensure .chromadb is in publichealth
        collection_name="publichealth",
    )


if __name__ == "__main__":
    pdf_urls = [
        {
            "title": "Introduction to Public Health",
            "url": "https://www.cdc.gov/training-publichealth101/media/pdfs/introduction-to-public-health.pdf",
        },
        {
            "title": "Blood lead levels of 4-11-year-old Mexican American, Puerto Rican, and Cuban children.",
            "url": "https://stacks.cdc.gov/view/cdc/63499/cdc_63499_DS1.pdf",
        },
        {
            "title": "Lead-contaminated imported tamarind candy and children's blood lead levels.",
            "url": "https://stacks.cdc.gov/view/cdc/64943/cdc_64943_DS1.pdf",
        },
        {
            "title": "Underreporting of minority AIDS deaths in San Francisco Bay area, 1985-86.",
            "url": "https://stacks.cdc.gov/view/cdc/63501/cdc_63501_DS1.pdf",
        },
        {
            "title": "Dental decay rates among children of migrant workers in Yakima, WA.",
            "url": "https://stacks.cdc.gov/view/cdc/63526/cdc_63526_DS1.pdf",
        },
        {
            "title": "Determinants of breast cancer screening among inner-city Hispanic women in comparison with other inner-city women.",
            "url": "https://stacks.cdc.gov/view/cdc/64233/cdc_64233_DS1.pdf",
        },
        {
            "title": "Measles reporting completeness during a community-wide epidemic in inner-city Los Angeles.",
            "url": "https://stacks.cdc.gov/view/cdc/64174/cdc_64174_DS1.pdf",
        },
    ]

    embed(pdf_urls)
