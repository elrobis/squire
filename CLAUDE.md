# Claude Code Instructions for Starfire Project

## Project Overview
AI-powered pipeline for processing D&D audio sessions into transcripts, speaker diarization, character attribution, and summaries.

## Session Status Tracking

### Command: "please create a status entry"

When this command is given:

1. **Create status file** in `status/` directory using pattern: `YYYY-MM-DD-#.md`
   - Date: Current date (YYYY-MM-DD format)
   - Session number: Start at 1, increment for multiple sessions same day
   - Format: Markdown (.md)

2. **File content** should include:
   - Brief summary of work completed this session
   - Key changes/improvements made
   - Files modified
   - Any issues encountered or resolved
   - Next steps or items left for future sessions

3. **Check existing files** in status/ to determine correct session number for the day

### Example filename: `2025-10-05-1.md` (first session of October 5th, 2025)

## Development Notes

- Virtual environment: `~/starfire_venv/`
- Main workflow: transcribe → diarize → align → attribute → summarize → index
- Uses Ollama with `gpt-oss:20b` model for LLM tasks
- Requires HuggingFace token for pyannote speaker diarization
- Audio files should be 16kHz mono WAV format