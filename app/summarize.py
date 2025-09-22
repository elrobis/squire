from pathlib import Path
import orjson, json, requests
from dotenv import dotenv_values
from app.prompts import SUMMARY_PROMPT

CFG = dotenv_values()

def _ollama(text: str):
    url = f"{CFG.get('OLLAMA_HOST','http://127.0.0.1:11434')}/api/generate"
    payload = {"model": CFG.get("OLLAMA_MODEL","gpt-oss:20b"),
               "prompt": text, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]

def _lines_to_scenes(lines, chunk_sec=480):
    scenes, cur, t0 = [], [], None
    for ln in lines:
        if t0 is None: t0 = ln["start"]
        cur.append(ln)
        if ln["end"] - t0 >= chunk_sec:
            scenes.append(cur); cur, t0 = [], None
    if cur: scenes.append(cur)
    return scenes

def summarize_session(session_id: str) -> Path:
    aligned = orjson.loads(Path(f"data/aligned/{session_id}.json").read_bytes())
    scenes = _lines_to_scenes(aligned["lines"], int(CFG.get("CHUNK_SEC","480")))
    out_dir = Path("data/summaries"); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{session_id}.jsonl"; out.write_text("", encoding="utf-8")

    for scene in scenes:
        text = "\n".join([f'{l["speaker"]}: {l["text"]}' for l in scene])
        prompt = SUMMARY_PROMPT.format(scene=text)
        resp = _ollama(prompt)
        # repair JSON if needed
        try:
            js = json.loads(resp)
        except Exception:
            start = resp.find('{'); end = resp.rfind('}')
            js = json.loads(resp[start:end+1])
        with out.open("a", encoding="utf-8") as f:
            f.write(orjson.dumps(js).decode()+"\n")
    return out