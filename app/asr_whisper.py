from faster_whisper import WhisperModel
from pathlib import Path
import orjson, os
from dotenv import dotenv_values

CFG = dotenv_values()

def transcribe_file(audio_path: Path) -> Path:
    audio_path = Path(audio_path)
    out_dir = Path("data/transcripts"); out_dir.mkdir(parents=True, exist_ok=True)

    # Try configurations with word timestamps (keeping speed where possible)
    configs = [
        # First try: int8_float16 with word timestamps (fastest, but may fail on some GPUs)
        CFG.get("WHISPER_COMPUTE", "int8_float16"),
        # Fallback: float16 with word timestamps (more compatible)
        "float16",
    ]

    segments, info = None, None
    last_error = None

    for i, compute_type in enumerate(configs):
        try:
            print(f"Attempting transcription with compute_type={compute_type}, word_timestamps=True")

            model = WhisperModel(
                CFG.get("WHISPER_MODEL","large-v3"),
                device="cuda",
                compute_type=compute_type,
            )

            segments, info = model.transcribe(
                str(audio_path),
                vad_filter=True,
                beam_size=5,
                word_timestamps=True
            )

            # Test if we can actually iterate through segments (where cuBLAS error occurs)
            test_segments = []
            for s in segments:
                test_segments.append({
                    "start": float(s.start), "end": float(s.end), "text": s.text,
                    "words": [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in (s.words or [])]
                })

            # If we get here, it worked!
            print(f"✓ Transcription successful with {compute_type}")
            segments = test_segments
            break

        except RuntimeError as e:
            last_error = e
            if "cuBLAS" in str(e) or "CUBLAS" in str(e):
                print(f"✗ cuBLAS error with {compute_type}, trying next configuration...")
                continue
            else:
                # Some other error, re-raise it
                raise
        except Exception as e:
            last_error = e
            print(f"✗ Error with {compute_type}: {e}")
            continue

    if segments is None:
        raise RuntimeError(f"All transcription configurations failed. Last error: {last_error}")

    # segments is now already a list of dicts from the successful test
    segs = segments

    session_id = audio_path.stem
    json_path = out_dir / f"{session_id}.json"
    txt_path  = out_dir / f"{session_id}.txt"

    json_path.write_bytes(orjson.dumps({"session": session_id, "segments": segs}, option=orjson.OPT_INDENT_2))
    txt_path.write_text("\n".join([s["text"] for s in segs]), encoding="utf-8")
    return json_path