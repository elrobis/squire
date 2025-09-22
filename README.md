# AI Audio Processing Pipeline

An AI-powered pipeline for processing audio files to extract dialogue, perform speaker diarization, character attribution, and generate summaries.

## Features

- **Audio Transcription**: Convert audio files to text using Whisper
- **Speaker Diarization**: Identify different speakers in audio
- **Dialogue Alignment**: Merge transcripts with speaker information
- **Character Attribution**: Map dialogue lines to characters using LLM
- **Summarization**: Generate scene summaries and beat sheets
- **Vector Search**: Index and query processed content

## Project Structure

```
├─ README.md
├─ .env                         # tokens + config
├─ requirements.txt
├─ data/
│  ├─ audio/                    # drop WAV/MP3 here
│  ├─ transcripts/              # whisper JSON + TXT
│  ├─ diarization/              # speaker turns (RTTM/JSON)
│  ├─ aligned/                  # transcript merged with speakers
│  ├─ attributed/               # character-attributed dialogue
│  └─ summaries/                # scene summaries/beat sheets
├─ chroma/                      # vector store
├─ app/
│  ├─ cli.py                    # Typer CLI entrypoint
│  ├─ asr_whisper.py            # transcription
│  ├─ diarize.py                # speaker diarization
│  ├─ align.py                  # align ASR segments ↔ speakers
│  ├─ attribute.py              # map lines to Characters via LLM
│  ├─ summarize.py              # scene/episode summaries
│  ├─ embed_index.py            # Chroma ingest + query
│  ├─ prompts.py                # prompt templates
│  └─ utils.py                  # ffmpeg, io helpers, chunking
```

## Setup

1. Activate the virtual environment:
   ```bash
   source ~/starfire_venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`

4. Place audio files in `data/audio/`

## Usage

Run the CLI tool:
```bash
python app/cli.py --help
```