# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based meeting minutes generator that uses OpenRouter API to transcribe audio/video files and generate structured meeting minutes. The project leverages OpenRouter's audio-capable chat models for transcription and LLMs for summarization.

## Development Commands

### Environment Setup
- `make venv` - Create Python virtual environment (.venv)
- `make install` - Set up base environment (minimal dependencies)
- `make install-whisper` - Install optional Whisper for local transcription

### Running the Application
- `make minutes FILE=Session.m4a OUT=minutes.txt` - Generate minutes using OpenRouter (recommended)
- `make transcript-only FILE=Session.m4a OUT=transcript.txt` - Transcription only, no minutes generation
- `make minutes-only TRANSCRIPT=transcript.txt OUT=minutes.txt` - Minutes from existing transcript
- `make minutes-enhanced TRANSCRIPT=transcript.txt OUT=enhanced.txt` - Contextually enriched minutes with web research
- `make transcript FILE=meeting.mp4 OUT=minutes.txt` - Transcribe + summarize with custom args
- `make run ARGS="..."` - Run script with custom arguments

### Environment Variables Required
- `OPENROUTER_API_KEY` - Required for OpenRouter transcription and summarization
- `OPENAI_API_KEY` - Optional, for OpenAI transcription fallback
- Optional headers: `OPENROUTER_APP_TITLE`, `OPENROUTER_SITE_URL`

### Cleanup
- `make clean` - Remove virtual environment

## Architecture

### Core Components
- **scripts/meeting_minutes.py** - Main application script (lines 1-886)
  - Handles multiple transcription backends: OpenRouter, OpenAI, local Whisper
  - Implements chunked processing for large files
  - Supports both map-reduce and full transcript synthesis strategies

### Key Features
- **Multi-backend transcription**: Auto-selects between OpenRouter (preferred), OpenAI, or local Whisper
- **Audio processing**: Uses ffmpeg to split large audio files into segments for chat-based STT
- **Smart model selection**: Cost-effective model selection for chunking vs final synthesis
- **Synthesis strategies**:
  - "full" - Single pass over entire transcript (better context)
  - "map-reduce" - Chunk summaries then synthesize (cheaper for large files)
  - "auto" - Automatically chooses based on size

### Environment Configuration
- Uses .env file loading (scripts/meeting_minutes.py:725-743)
- Supports .env and .env.local files
- Environment variables take precedence over .env files

## Model Configuration

### OpenRouter Models
- Default STT chat model: `google/gemini-2.5-flash-lite`
- Configurable via `STT_CHAT_MODEL` environment variable
- Alternative models: `openai/gpt-4o-audio-preview`
- Cost optimization: cheap models for chunking, stronger for final synthesis

### Transcription Options
- OpenRouter chat-based STT (segments audio for chat/completions with input_audio)
- OpenRouter STT endpoint (if available on account)
- OpenAI Whisper API
- Local Whisper (requires `make install-whisper`)

## File Structure
- `scripts/` - Contains the main meeting_minutes.py application
- `.venv/` - Python virtual environment (created by make venv)
- `.env` - Environment configuration (not tracked, contains API keys)
- `Makefile` - Build and run automation
- `AGENTS.md` - Development guidelines and coding standards

## Development Notes
- Python 3.11+ required
- ffmpeg dependency for audio processing
- No external Python package dependencies for core functionality (uses only urllib and json)
- OpenRouter integration uses raw HTTP requests, no SDK dependency
- Supports various audio/video formats through ffmpeg