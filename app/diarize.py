from pathlib import Path
import orjson, os
from dotenv import dotenv_values
from pyannote.audio import Pipeline

CFG = dotenv_values()

def diarize_file(audio_path: Path) -> Path:
    out_dir = Path("data/diarization"); out_dir.mkdir(parents=True, exist_ok=True)
    session_id = Path(audio_path).stem

    pipeline = Pipeline.from_pretrained(
        CFG.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1"),
        use_auth_token=CFG.get("HF_TOKEN")
    )
    diarization = pipeline(str(audio_path))

    # Save RTTM-like JSON
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start), "end": float(turn.end),
            "speaker": speaker
        })

    json_path = out_dir / f"{session_id}.json"
    json_path.write_bytes(orjson.dumps({"session": session_id, "turns": turns}, option=orjson.OPT_INDENT_2))
    return json_path