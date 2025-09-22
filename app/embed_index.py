from pathlib import Path
import chromadb, orjson
from chromadb.utils import embedding_functions

def ingest_session(session_id: str):
    client = chromadb.PersistentClient(path="chroma")
    coll = client.get_or_create_collection(name="starfire")

    # simple bag-of-words embedding is weak; use OpenAI or local if you prefer.
    # Placeholder: use Chroma's default embedder if present, else text-as-id.
    # For a strong local option, consider "bge-small-en" via SentenceTransformers.

    docs, metas, ids = [], [], []
    # transcripts
    aligned = orjson.loads(Path(f"data/aligned/{session_id}.json").read_bytes())
    for i, ln in enumerate(aligned["lines"]):
        docs.append(f'{ln["speaker"]}: {ln["text"]}')
        metas.append({"session": session_id, "type":"line", "start": ln["start"], "end": ln["end"]})
        ids.append(f"{session_id}-line-{i}")

    # summaries
    summ_path = Path(f"data/summaries/{session_id}.jsonl")
    if summ_path.exists():
        for i, line in enumerate(summ_path.read_text().splitlines()):
            js = orjson.loads(line)
            docs.append(js.get("summary",""))
            metas.append({"session": session_id, "type":"summary", "idx": i})
            ids.append(f"{session_id}-sum-{i}")

    coll.add(documents=docs, metadatas=metas, ids=ids)