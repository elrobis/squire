from pathlib import Path
import orjson, json, os
from dotenv import dotenv_values
from app.prompts import ATTRIBUTION_PROMPT
import requests
import time
from datetime import datetime
from requests.exceptions import ReadTimeout

CFG = dotenv_values()

def _split_chunk(chunk: list) -> tuple[list, list]:
    """Split a chunk into two roughly equal halves by line count."""
    mid = len(chunk) // 2
    return chunk[:mid], chunk[mid:]

def _process_chunk_with_retry(chunk: list, roster: str, chunk_num: int, total_chunks: int, max_splits: int = 2) -> list:
    """Process a chunk with automatic splitting on timeout."""
    current_chunks = [chunk]
    split_level = 0

    while current_chunks and split_level <= max_splits:
        next_chunks = []
        all_results = []

        for i, ch in enumerate(current_chunks):
            if not ch:  # Skip empty chunks
                continue

            chunk_label = f"{chunk_num}" if len(current_chunks) == 1 else f"{chunk_num}.{i+1}"

            try:
                start_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing chunk {chunk_label}/{total_chunks} ({len(ch)} lines, {len(orjson.dumps(ch))} chars)...")

                prompt = ATTRIBUTION_PROMPT.format(
                    roster=roster,
                    lines=orjson.dumps(ch).decode()
                )
                resp = _ollama(prompt)
                elapsed = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Chunk {chunk_label} completed successfully in {elapsed:.1f}s")

                # Parse response
                try:
                    block = json.loads(resp)
                    all_results.extend(block)
                except Exception:
                    # tiny repair: look for first [ ... ] in text
                    start = resp.find('['); end = resp.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        try:
                            block = json.loads(resp[start:end+1])
                            all_results.extend(block)
                        except Exception:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Failed to parse LLM response for chunk {chunk_label}")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: No valid JSON found in response for chunk {chunk_label}")

            except ReadTimeout:
                if split_level < max_splits and len(ch) > 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout on chunk {chunk_label}, splitting into smaller pieces...")
                    left, right = _split_chunk(ch)
                    next_chunks.extend([left, right])
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout on chunk {chunk_label}, cannot split further - skipping")

        if next_chunks:
            current_chunks = next_chunks
            split_level += 1
        else:
            break

    return all_results

def _ollama(prompt: str):
    url = f"{CFG.get('OLLAMA_HOST','http://127.0.0.1:11434')}/api/generate"
    payload = {"model": CFG.get("OLLAMA_MODEL","gpt-oss:20b"),
               "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=1200)
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
        if cur_len + len(s) > 3000:  # conservative token-safe chunking
            chunks.append(cur); cur, cur_len = [], 0
        cur.append(ln); cur_len += len(s)
    if cur: chunks.append(cur)

    out_dir = Path("data/attributed"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.jsonl"
    checkpoint_path = out_dir / f"{session_id}.checkpoint"

    # Check for existing progress
    start_chunk = 1
    if checkpoint_path.exists() and out_path.exists():
        try:
            checkpoint_data = orjson.loads(checkpoint_path.read_bytes())
            start_chunk = checkpoint_data.get("last_completed_chunk", 0) + 1
            if start_chunk <= len(chunks):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming from chunk {start_chunk} (found checkpoint)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] All chunks already completed!")
                return out_path
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not read checkpoint, starting fresh: {e}")
            start_chunk = 1
            out_path.write_text("", encoding="utf-8")
    else:
        # Fresh start
        out_path.write_text("", encoding="utf-8")
        checkpoint_path.write_text(orjson.dumps({"last_completed_chunk": 0}).decode(), encoding="utf-8")

    total_chunks = len(chunks)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing chunks {start_chunk}-{total_chunks} for attribution...")

    for i in range(start_chunk - 1, len(chunks)):
        ch = chunks[i]
        chunk_num = i + 1

        # Process chunk with automatic retry/splitting on timeout
        block = _process_chunk_with_retry(ch, roster, chunk_num, total_chunks)

        if block:
            # Write results to output file
            with out_path.open("a", encoding="utf-8") as f:
                for item in block:
                    f.write(orjson.dumps(item).decode() + "\n")

        # Update checkpoint after processing (successful or not)
        checkpoint_data = {"last_completed_chunk": chunk_num, "timestamp": datetime.now().isoformat()}
        checkpoint_path.write_text(orjson.dumps(checkpoint_data).decode(), encoding="utf-8")

    # Clean up checkpoint file when all chunks are complete
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All chunks completed successfully!")
    return out_path