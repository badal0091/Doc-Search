# DocSearch

DocSearch is a Gen AI RAG search app deployed at <https://gramener.com/docsearch/>.

## Features

## Setup

1. Clone this repo and change directory to it
2. Create a `.secrets.yaml` with `DOCSEARCH_LLMPROXY_JWT` set to your [LLM Foundry token](https://llmfoundry.straive.com/code)
3. Install [uv](https://github.com/astral-sh/uv)
4. Run `uvx --python 3.9 --with-requirements requirements.txt gramex` to run the app

## Add a demo

**Update [`config.yaml`](config.yaml)** to include a `newdemo` under `demos`.

```yaml
# config.yaml
demos:
  newdemo: # The key MUST be the same as the directory name
    title: Your demo's title
    body: 1-sentence description of the demo
    questions: # Pick 3+ good questions
      - ...
      - ...
      - ...
    # The rest of these are optional

    # To restrict the demo to users with emails from specific domains
    domains: [gramener.com, straive.com, learningmate.com]

    # If clicking on the URL should open a specific page, add the URL here.
    # If you use the 🔑 symbol in the URL, it will be replaced by the "key" field
    link: https://www.google.com/search?q=🔑

    # Optional name of client. Just for reference
    client: Client name

    # Show a warning if results have lower similarity than this
    min_similarity: 0.3

    # Add filters. Each filter is a key-value pair. The key is the field name
    filters:
      key_in_chroma_db:
        label: Display label in the UI
        required: false # If true, the user must select a value for this filter
      decision:
        label: Decision
        required: true

    # How to color the nodes in the graph. If not specified, nodes are colored by relevance
    color:
      field: key_in_chroma_db # Field to color by, e.g. decision
      values: { Accept: green, Reject: red } # If decision=Accept, color it green, etc.
      relevanceOpacity: true # If true, lower relevance values are transparent
```

- **Create `newdemo/scrape.py`**.
  - The directory name `newdemo` should match the key in config.yaml under `demos:`
  - It should create `newdemo/.chromadb/` with all document embeddings.
  - The collection **MUST** be named `newdemo`.
  - The metadata **MUST** include a `key` (unique identifier) and `h1` (title).
  - See examples of `scrape.py`:
    - [`learningmatepolicies/scrape.py`](learningmatepolicies/scrape.py)
    - [`paperrejections/scrape.py`](paperrejections/scrape.py)
- **Run `newdemo/scrape.py` once**. It will create a `.chromadb` folder with the embeddings.
- **Use Git LFS** to commit `newdemo/.chromadb/`

**Do not commit source documents** unless there is a single document under 2MB.

## How this app was built

[![Explainer video](https://img.youtube.com/vi/fAm67WNZ3F0/0.jpg)](https://youtu.be/fAm67WNZ3F0)
# DocSearch
