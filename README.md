# AI Audio Processing Pipeline

An AI-powered pipeline for processing audio files to extract dialogue, perform speaker diarization, character attribution, and generate summaries.

Created to extract dialog and summarize Dungeons & Dragons gameplay sessions, but applicable to similarly-themed workflows.

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

### Prerequisites

1. **FFmpeg** (required for audio processing):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg
   ```

2. **Ollama** (required for LLM inference):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull the model (20B parameter model recommended)
   ollama pull gpt-oss:20b
   ```

3. **Hugging Face Token** (required for speaker diarization):
   - Create account at https://huggingface.co
   - Get a read-only API token from your settings
   - Accept the license for `pyannote/speaker-diarization-3.1`

### Installation

- Establish a virtual environment:
   ```bash
   python3 -m venv ~/starfire_venv
   source ~/starfire_venv/bin/activate
   pip install -r requirements.txt
   ```

- Copy `.env-default` into a new file, `.env`, then configure environment variables:
   ```bash
   cp .env-default .env
   # Edit .env and set your HF_TOKEN
   ```

- Place audio files in `data/audio/`

### Audio Preparation

If you have MP3 files, convert them to the required WAV format:

```bash
# Batch convert all MP3 files to WAV (16kHz mono)
for f in data/audio/*.mp3; do
  ffmpeg -i "$f" -ac 1 -ar 16000 -c:a pcm_s16le "${f%.mp3}.wav"
done
```

## Usage

Run the CLI tool:
```bash
python app/cli.py --help
```

### Complete Workflow

Process a D&D session from start to finish:

```bash
# 1. Transcribe audio to text
python -m app.cli transcribe data/audio/Session_090123_01.wav

# 2. Identify speakers in the audio
python -m app.cli diarize data/audio/Session_090123_01.wav

# 3. Align transcript with speaker information
python -m app.cli align Session01
```

Create a `roster.json` file to map speakers to characters:
```json
{
  "dm": "Luke (DM)",
  "players": [
    {"name": "Jerome", "character": "Aguiar", "notes": "Human fighter"},
    {"name": "Nancy", "character": "Juniper", "notes": "Elf magic user"},
    {"name": "Chris", "character": "Starble", "notes": "Dwarf fighter"}
  ],
  "known_npcs": [
    {"name":"Glade", "notes":"Member of the party but controlled by the DM"},
    {"name":"Starla", "notes":"High charisma, older human female, innate storyteller character, lives outside of Drexville"}
  ],
  "tone":"Low magic, survival-focused campaign"
}
```

```bash
# 4. Attribute dialogue lines to characters
python -m app.cli attribute Session01 roster.json

# 5. Generate scene summaries
python -m app.cli summarize Session01

# 6. Index content for vector search
python -m app.cli index Session01
```