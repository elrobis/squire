import typer
from pathlib import Path
from dotenv import load_dotenv

from app.asr_whisper import transcribe_file
from app.align import align_asr_speakers
from app.attribute import attribute_characters
from app.summarize import summarize_session
from app.embed_index import ingest_session

app = typer.Typer(help="Starfire pipeline CLI")

@app.command()
def transcribe(audio_path: Path):
    out = transcribe_file(audio_path)
    typer.echo(f"Transcript saved: {out}")

@app.command()
def diarize(audio_path: Path):
    # <-- lazy import so transcribe doesn't pull pyannote
    from app.diarize import diarize_file
    out = diarize_file(audio_path)
    typer.echo(f"Diarization saved: {out}")

@app.command()
def align(session_id: str):
    out = align_asr_speakers(session_id)
    typer.echo(f"Aligned JSON: {out}")

@app.command()
def attribute(session_id: str, roster_path: Path):
    out = attribute_characters(session_id, roster_path)
    typer.echo(f"Attributed dialogue: {out}")

@app.command()
def summarize(session_id: str):
    out = summarize_session(session_id)
    typer.echo(f"Summaries: {out}")

@app.command()
def index(session_id: str):
    ingest_session(session_id)
    typer.echo("Indexed to Chroma.")

if __name__ == "__main__":
    load_dotenv()
    app()