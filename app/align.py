from pathlib import Path
import orjson

def align_asr_speakers(session_id: str) -> Path:
    asr = orjson.loads(Path(f"data/transcripts/{session_id}.json").read_bytes())
    dia = orjson.loads(Path(f"data/diarization/{session_id}.json").read_bytes())

    aligned = []
    turns = dia["turns"]
    for seg in asr["segments"]:
        mid = 0.5*(seg["start"] + seg["end"])
        # naive: find speaker turn containing segment midpoint
        spk = next((t["speaker"] for t in turns if t["start"] <= mid <= t["end"]), "SPK_UNK")
        aligned.append({
            "start": seg["start"], "end": seg["end"],
            "speaker": spk, "text": seg["text"], "words": seg.get("words", [])
        })

    out_dir = Path("data/aligned"); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{session_id}.json"
    out.write_bytes(orjson.dumps({"session": session_id, "lines": aligned}, option=orjson.OPT_INDENT_2))
    return out