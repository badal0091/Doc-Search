"""Handles search for the docsearch app"""

import gramex
import gramex.cache
import json
import numpy as np
import os
import sqlite3
import tornado.httpclient
from gramex.handlers import BaseHandler
from gramex.transforms import handler
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tornado.gen import coroutine


folder = os.path.dirname(os.path.abspath(__file__))
secrets = gramex.cache.open(".secrets.yaml", rel=True)
file_store = LocalFileStore(os.path.join(folder, ".embeddings"))
base = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://gramener.com/llmproxy/v1/",
    openai_api_key=f'{secrets["DOCSEARCH_LLMPROXY_JWT"]}:docsearch',
)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)
dbs = {}


def get_db(app):
    if app not in dbs:
        dbs[app] = Chroma(
            persist_directory=os.path.join(folder, app, ".chromadb"),
            embedding_function=cached_embedder,
            collection_name=app,
        )
    return dbs[app]


@coroutine
@handler
def similarity(handler, app: str, q: str, k: int = 50):
    db = get_db(app)
    conf = gramex.cache.open("config.yaml", rel=True)["demos"][app]
    # Get all applicable non-empty filters
    filters = {
        key: handler.args[key][0]
        for key in conf.get("filters", [])
        if key in handler.args and len(handler.args[key]) and handler.args[key][0]
    }
    print(f"Filters applied: {len(filters)} - {filters}")
    
    docs = yield gramex.service.threadpool.submit(
        db.similarity_search_with_score, q, k=k, filter=filters
    )
    print(f"Number of results before sorting: {len(docs)}")
    docs = sorted(docs, key=lambda doc: doc[1], reverse=True)
    print(f"Number of results after sorting: {len(docs)}")
    embeddings = np.array(cached_embedder.embed_documents([d.page_content for d, _ in docs]))
    similarity = np.dot(embeddings, embeddings.T)
    return {
        "matches": [dict(doc, score=score) for doc, score in docs],
        "similarity": similarity.tolist(),
    }


def filter_values(app, field):
    conn = sqlite3.connect(os.path.join(folder, app, ".chromadb", "chroma.sqlite3"))
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key=?", (field,))
    return [row[0] for row in cursor.fetchall()]


class Summarizer(BaseHandler):
    """Summarizes a context"""

    async def post(self):
        body = json.loads(self.request.body)
        config = gramex.cache.open("config.yaml", rel=True)
        prompt = f"""{config['prompts']['answer']}

Tone: {body['Tone']}. {config['styles']['Tone'][body['Tone']]}
Format: {body['Format']}. {config['styles']['Format'][body['Format']]}
Language: {body['Language']}. {config['styles']['Language'][body['Language']]}
"""
        if body.get("Followup"):
            prompt += config["prompts"]["followup"]
        message = f"```context\n${body['context']}\n```\n\nQuestion: ${body['q']}"

        api = {
            "request": "https://llmfoundry.straive.com/openai/v1/chat/completions",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {secrets["DOCSEARCH_LLMPROXY_JWT"]}:docsearch',
            },
            "body": json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "stream": True,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": message},
                    ],
                }
            ),
        }
        http_client = tornado.httpclient.AsyncHTTPClient()
        # TODO: Use headers to catch errors
        await http_client.fetch(**api, streaming_callback=self.send, request_timeout=300)

    def send(self: "Summarizer", data: bytes):
        """Write data to response and flush it"""
        self.write(data)
        self.flush()
