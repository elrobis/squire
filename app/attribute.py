from pathlib import Path
import orjson, json, os
from dotenv import dotenv_values
from app.prompts import ATTRIBUTION_PROMPT
import requests

CFG = dotenv_values()

def _ollama(prompt: str):
    url = f"{CFG.get('OLLAMA_HOST','http://127.0.0.1:11434')}/api/generate"
    payload = {"model": CFG.get("OLLAMA_MODEL","gpt-oss:20b"),
               "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]

def attribute_characters(session_id: str, roster_path: Path) -> Path:
    aligned = orjson.loads(Path(f"data/aligned/{session_id}.json").read_bytes())
    roster = roster_path.read_text()
    # keep chunks small to keep context tight
    lines = aligned["lines"]
    chunks = []
    cur, cur_len = [], 0
    for ln in lines:
        s = orjson.dumps(ln).decode()
        if cur_len + len(s) > 6000:  # conservative token-safe chunking
            chunks.append(cur); cur, cur_len = [], 0
        cur.append(ln); cur_len += len(s)
    if cur: chunks.append(cur)

    out_dir = Path("data/attributed"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.jsonl"
    out_path.write_text("", encoding="utf-8")

    for ch in chunks:
        prompt = ATTRIBUTION_PROMPT.format(
            roster=roster,
            lines=orjson.dumps(ch).decode()
        )
        resp = _ollama(prompt)
        # be forgiving of minor JSON issues
        try:
            block = json.loads(resp)
        except Exception:
            # tiny repair: look for first [ ... ] in text
            start = resp.find('['); end = resp.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    block = json.loads(resp[start:end+1])
                except Exception:
                    print(f"Warning: Failed to parse LLM response, skipping chunk. Response preview: {resp[:200]}...")
                    continue
            else:
                print(f"Warning: No valid JSON array found in LLM response, skipping chunk. Response preview: {resp[:200]}...")
                continue
        with out_path.open("a", encoding="utf-8") as f:
            for item in block:
                f.write(orjson.dumps(item).decode() + "\n")
    return out_path