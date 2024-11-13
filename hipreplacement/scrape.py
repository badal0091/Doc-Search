# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
# ]
# ///
# Download these videos via `yt-dlp`:
#   yt-dlp https://www.youtube.com/watch?v=5NqJa_J2dfw
#   yt-dlp https://www.youtube.com/watch?v=TQegBCVcOKo
#   yt-dlp https://www.youtube.com/watch?v=TQegBCVcOKo
#   yt-dlp https://www.youtube.com/watch?v=XB2cUiSj_9I
#   yt-dlp https://www.youtube.com/watch?v=rhQMjdiHWJw
#
# Then convert to frames via ffmpeg. Fit to a max height of 512 px.
#
# for file in *.webm *.mkv *.mp4; do
#     filename="${file%.*}"
#     ffmpeg -i "$file" -vf "select='key',showinfo,scale='if(gt(a,1),512,-2)':'if(gt(a,1),-2,512)'" -vsync vfr -compression_level 10 "$filename-%03d.jpg" 2>&1 | grep 'pts_time' | sed 's/.*pts_time:\([0-9.]*\).*/\1/' > "$filename-timestamps.txt"
#     whisper --language en --output_dir . --output_format json "$file"
# done
# Then run scrape.py

import base64
import httpx
import json
import os
import urllib.parse
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
        with open(f"{prefix}.json") as f:
            transcript = json.load(f)["segments"]

        for i in range(0, len(group), 10):
            chunk = group[i : i + 10]
            start, end = i, i + len(chunk) - 1
            print(f"Processing {prefix} {start}-{end}...")  # noqa
            start_time, end_time = timestamps[start], timestamps[end]
            # Get .text from transcript where .start, .end intersect with start_time, end_time
            chunk_transcript = " ".join(
                [
                    segment["text"]
                    for segment in transcript
                    if segment["start"] <= start_time <= segment["end"]
                    or segment["start"] <= end_time <= segment["end"]
                ]
            )
            print(chunk_transcript)  # noqa
            properties = schema["properties"]["frames"]["items"]["properties"]
            properties["frame"]["description"] = f"Frame numbers from {start} to {end}"
            body = {
                "model": "gpt-4o-mini",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "frames",
                            "description": "Describe each frame",
                            "parameters": schema,
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "frames"}},
                "messages": [
                    {
                        "role": "system",
                        "content": f"""
These are frames {start}-{end} (out of {len(group)}) from a SAFE INSTRUCTIONAL video. Transcript below:

{chunk_transcript}

Describe each frame for a blind person calling `frames([frame, frame, ...])`.
""".strip(),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                    + base64.b64encode(open(image, "rb").read()).decode("utf-8"),
                                    "detail": "low",
                                },
                            }
                            for image in chunk
                        ],
                    },
                ],
            }
            response = httpx.post(
                "https://llmfoundry.straive.com/openai/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=300,
            )
            function = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
            input = json.loads(function["arguments"])["frames"]
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
        with open(f"{prefix}.json") as f:
            transcript = json.load(f)["segments"]
        # Get the part between [...] inside prefix
        title = prefix.split("[")[0].strip()
        id = prefix.split("[")[1].split("]")[0]
        for frame in framelist:
            chunk_transcript = " ".join(
                [
                    segment["text"]
                    for segment in transcript
                    if segment["start"] <= frame["timestamp"] <= segment["end"]
                ]
            )
            image = urllib.parse.quote(f'{prefix}-{frame["frame"]:03d}.jpg')
            documents.append(
                Document(
                    page_content=f"""
[![]({image})](https://youtu.be/{id}?t={int(frame['timestamp'])})

At: {frame["timestamp"]} seconds

{frame['description']}

**Objects**: {', '.join(frame['objects'])}

**Transcript**: {chunk_transcript}
""".strip(),
                    metadata={"h1": title, "key": id},
                )
            )

    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory=".chromadb",
        collection_name="hipreplacement",
    )


if __name__ == "__main__":
    embed()
