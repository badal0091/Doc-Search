# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
# ]
# ///
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


urls = {
    "Law No. (4) of 2022 Regulating Virtual Assets in the Emirate of Dubai": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_338_VER1.pdf",
    "Cabinet Decision No. 111/2022 On the Regulation of Virtual Assets and Their Service Providers": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_339_VER1.pdf",
    "Cabinet Decision No. 112/2022 On Delegating Certain Competencies related to the Regulation of Virtual Assets": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_340_VER1.pdf",
    "Virtual Assets and Related Activities Regulations 2023": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_18_VER992_0.pdf",
    "Marketing Regulations: Administrative Order No. 01/2022": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_341_VER1.pdf",
    "Marketing Regulations: Administrative Order No. 02/2022": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_342_VER1.pdf",
    "Administration Resolution No. (3) of 2023 – Grievance Committee": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_345_VER2.pdf",
    "Company Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_36_VER984.pdf",
    "Compliance and Risk Management Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_123_VER1.pdf",
    "Technology and Information Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_169_VER1.pdf",
    "Market Conduct Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_190_VER1.pdf",
    "Advisory Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_215_VER1.pdf",
    "Broker-Dealer Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_226_VER1.pdf",
    "Custody Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_243_VER1094_0.pdf",
    "Exchange Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_258_VER1.pdf",
    "Lending and Borrowing Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_279_VER1.pdf",
    "VA Management and Investment Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_317_VER1.pdf",
    "VA Transfer and Settlement Services Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_347_VER1.pdf",
    "Virtual Asset Issuance Rulebook": "https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_293_VER2.pdf",
}

papers_file = "arxiv-metadata-oai-snapshot.parquet"
# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore('.embeddings')
base = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)


def embed_files(urls, app):
    docs = []
    for title, url in urls.items():
        # Get the file path (PDF file name) from the URL
        filename = url.split("/")[-1]
        print(title)  # noqa:T201
        if not os.path.exists(filename):
            r = requests.get(url, timeout=30)
            with open(filename, "wb") as f:
                f.write(r.content)
        loader = PyPDFLoader(filename)
        docparts = loader.load_and_split()
        for doc in docparts:
            doc.metadata['h1'] = f'{title} page {doc.metadata["page"]}'
            doc.metadata['key'] = filename
        docs += docparts
    return Chroma.from_documents(
        docs,
        cached_embedder,
        persist_directory='.chromadb',
        collection_name=app,
    )


if __name__ == "__main__":
    embed_files(urls, "vara")
