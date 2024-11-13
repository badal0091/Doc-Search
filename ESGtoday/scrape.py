# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "langchain-chroma",
#     "langchain-community",
#     "beautifulsoup4",
#     "requests",
#     "tqdm",
# ]
# ///

import os
import yaml
import requests
from bs4 import BeautifulSoup
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

def scrape_website(url):
    """Scrape content from a website and return the text."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, verify=False)  # Bypass SSL verification
    response.raise_for_status()  # Raise an error for bad responses
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text from paragraphs and headers
    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4'])
    return ' '.join([para.get_text() for para in paragraphs])

def embed(website_urls):
    docs, metadata = [], []
    for row in tqdm(website_urls):
        content = scrape_website(row["url"])
        key = row["url"]
        docs.append(content)
        metadata.append({"key": key, "title": row["title"]})
    
    documents = text_splitter.create_documents(docs, metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=os.path.join(current_dir, ".chromadb"),
        collection_name="ESGtoday",
    )

if __name__ == "__main__":
    website_urls = [
        {
            "title": "Miranda Lambert's MuttNation-Tractor Supply Relief for Rescues Fund Donates $250,000 to Support Hurricane Recovery",
            "url": "https://www.esgtoday.com/miranda-lamberts-muttnation-tractor-supply-relief-for-rescues-fund-donates-250000-to-support-hurricane-recovery/",
        },
        {
            "title": "Peloton Announces Fourth Annual ESG Report, Highlighting Progress on Impact Initiatives",
            "url": "https://www.esgtoday.com/peloton-announces-fourth-annual-esg-report-highlighting-progress-on-impact-initiatives/",
        },
        {
            "title": "Transition Industries Expands Public-Private Partnerships for Its Pacifico Mexinol Project in Sinaloa, Mexico",
            "url": "https://www.esgtoday.com/transition-industries-expands-public-private-partnerships-for-its-pacifico-mexinol-project-in-sinaloa-mexico/",
        },
        {
            "title": "Coty Achieves Key Milestones and Sets New Targets in FY24 Sustainability Report",
            "url": "https://www.esgtoday.com/coty-achieves-key-milestones-and-sets-new-targets-in-fy24-sustainability-report/",
        },
        {
            "title": "Penguin Solutions Releases 2023 Environmental, Social, and Governance Report",
            "url": "https://www.esgtoday.com/penguin-solutions-releases-2023-environmental-social-and-governance-report/",
        },
        {
            "title": "Pacific Premier Bancorp, Inc. Director Rose McKinney-James Honored With the 2024 Clean Energy Education & Empowerment Lifetime Achievement Award by the U.S. Department of Energy",
            "url": "https://www.esgtoday.com/pacific-premier-bancorp-inc-director-rose-mckinney-james-honored-with-the-2024-clean-energy-education-empowerment-lifetime-achievement-award-by-the-u-s-department-of-energy/",
        },
        {
            "title": "PUMA's 'Stitch + Spice' Running for Top Prize at the World's Biggest Sustainability Film Festival",
            "url": "https://www.esgtoday.com/pumas-stitch-spice-running-for-top-prize-at-the-worlds-biggest-sustainability-film-festival/",
        },
        {
            "title": "Nextracker Releases Inaugural Sustainability Report",
            "url": "https://www.esgtoday.com/nextracker-releases-inaugural-sustainability-report/",
        },
        {
            "title": "Six Global Environmental Organizations Unite to Scale Climate and Conservation Outcomes Through Sovereign Debt Conversions",
            "url": "https://www.esgtoday.com/six-global-environmental-organizations-unite-to-scale-climate-and-conservation-outcomes-through-sovereign-debt-conversions/",
        },
        {
            "title": "PUMA Part of Consortium to Unveil World's First Piece of 100% “Fibre-to-Fibre” Biorecycled Clothing",
            "url": "https://www.esgtoday.com/puma-part-of-consortium-to-unveil-worlds-first-piece-of-100-fibre-to-fibre-biorecycled-clothing/",
        },
    ]

    embed(website_urls) 