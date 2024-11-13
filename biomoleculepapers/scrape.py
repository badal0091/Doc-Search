# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "pandas",
#     "pyarrow",
#     "tiktoken",
#     "tqdm",
# ]
# ///
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import zipfile
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm import tqdm

papers_file = "arxiv-metadata-oai-snapshot.parquet"
# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore('.embeddings')
base = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)


def get_papers():
    """Save specific fields from the arXiv metadata to a parquet file"""
    source = "https://www.kaggle.com/datasets/Cornell-University/arxiv/"
    papers = "arxiv-metadata-oai-snapshot.json.zip"
    if not os.path.exists(papers):
        raise FileNotFoundError(f"Download {papers} from {source}")
    if not os.path.exists(papers_file):
        parquet_writer = None
        selected_fields = ["id", "categories", "title", "abstract", "update_date"]
        dtypes = {key: "str" for key in selected_fields}
        with zipfile.ZipFile(papers, "r") as z:
            with z.open("arxiv-metadata-oai-snapshot.json") as f:
                for chunk in tqdm(pd.read_json(f, lines=True, chunksize=100000, dtype=dtypes)):
                    table = pa.Table.from_pandas(chunk[selected_fields])
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(
                            papers_file, table.schema, compression="snappy"
                        )
                    parquet_writer.write_table(table)
        if parquet_writer:
            parquet_writer.close()


def embed_category(category, app):
    data = pd.read_parquet(papers_file)
    # Get all categories
    categories = data.categories.str.split().explode().value_counts()
    if category not in categories:
        raise ValueError(f"{category} not in {categories.index}")
    df = data[data.categories.str.contains(category, na=False)]
    df = df.rename(columns={"title": "h1", "id": "key"})[['key', 'h1', 'abstract']]
    return Chroma.from_documents(
        DataFrameLoader(df, page_content_column='abstract').load(),
        cached_embedder,
        persist_directory='.chromadb',
        collection_name=app,
    )


if __name__ == "__main__":
    get_papers()
    embed_category("q-bio.BM", "biomoleculepapers")
