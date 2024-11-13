# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "langchain",
#     "langchain-community",
#     "langchain-core",
# ]
# ///

# First, convert the videos into JPEGs with timestamps as follows
#   filename="${1%.*}"
#   ffmpeg -i "$1" -vf "select='key',showinfo,scale='if(gt(a,1),512,-2)':'if(gt(a,1),-2,512)'" -vsync vfr -compression_level 10 "$filename-%03d.jpg" 2>&1 | grep 'pts_time' | sed 's/.*pts_time:\([0-9.]*\).*/\1/' > "$filename-timestamps.txt"
# Then run scrape.py

import base64
import httpx
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore(".embeddings")
base = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)


def create_frames():
    files = [f for f in os.listdir(".") if f.endswith(".jpg")]

    # group files by prefix -- splitting the file by the last hyphen
    groups = {}
    for f in files:
        prefix = f.rsplit("-", 1)[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(f)

    headers = {"Authorization": f"Bearer {os.environ['LLMFOUNDRY_TOKEN']}:docsearch-videosearch"}

    schema = {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "description": "List of frames in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame": {
                            "type": "integer",
                            "description": "Frame number 0, 1, ...",
                        },
                        "objects": {
                            "type": "array",
                            "description": "List of objects in scene (mention ONLY changes from previous frame)",
                            "items": {
                                "type": "string",
                            },
                        },
                        "description": {
                            "type": "string",
                            "description": "Explanation for blind person (focus on changes from previous frame)",
                        },
                    },
                    "required": ["frame", "objects", "description"],
                },
            },
        },
    }

    # Loop through each group in chunks of at most 10 images each
    frames = {}
    for prefix, group in groups.items():
        with open(f"{prefix}-timestamps.txt") as f:
            timestamps = [float(line.strip()) for line in f.readlines()]

        for i in range(0, len(group), 10):
            chunk = group[i : i + 10]
            start, end = i, i + len(chunk) - 1
            print(f"Processing {prefix} {start}-{end}...")  # noqa
            properties = schema["properties"]["frames"]["items"]["properties"]
            properties["frame"]["description"] = f"Frame numbers from {start} to {end}"
            body = {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 4096,
                "tools": [
                    {
                        "name": "frames",
                        "description": "Describe each frame",
                        "input_schema": schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "frames"},
                "system": f"""
These are frames {start}-{end} (out of {len(group)} from a SAFE video.
Describe each frame for a blind person calling `frames([frame, frame, ...])`.
""".strip(),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64.b64encode(open(image, "rb").read()).decode(
                                        "utf-8"
                                    ),
                                },
                            }
                            for image in chunk
                        ],
                    },
                ],
            }
            response = httpx.post(
                "https://llmfoundry.straive.com/anthropic/v1/messages",
                json=body,
                headers=headers,
                timeout=300,
            )
            input = response.json()["content"][0]["input"]["frames"]
            for item in input:
                if item["frame"] < len(timestamps):
                    item["timestamp"] = timestamps[item["frame"]]
            frames.setdefault(prefix, []).extend(input)
            if len(input) != len(chunk):
                print(f"Mismatched lengths. Got {len(input)} not {len(chunk)}")  # noqa

    with open("frames.json", "w") as f:
        json.dump(frames, f, indent=2)


def embed():
    if not os.path.exists("frames.json"):
        create_frames()
    with open("frames.json") as f:
        frames = json.load(f)

    documents = []
    for prefix, framelist in frames.items():
        for frame in framelist:
            documents.append(
                Document(
                    page_content=f"""
[![]({prefix}-{frame["frame"]:03d}.jpg)]({prefix}.mp4#t={frame['timestamp']})

At: {frame["timestamp"]} seconds

{frame['description']}

**Objects**: {', '.join(frame['objects'])}
""".strip(),
                    metadata={"h1": prefix, "key": prefix},
                )
            )

    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="videosearch",
    )


if __name__ == "__main__":
    embed()
