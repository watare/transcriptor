# transcriptor

OpenRouter‑powered meeting minutes generator from local audio/video.

What it does
- Transcribes local files using OpenRouter audio‑capable chat models (or STT endpoint if available).
- Summarizes into clean, email‑ready minutes with subject + sections.

Quick start
- Requirements: Python 3.11+, ffmpeg, an OpenRouter API key.
- Setup:
  - make venv
  - export OPENROUTER_API_KEY=sk-or-...
- Run on a file (OpenRouter only):
  - make minutes FILE=Session.m4a OUT=minutes.txt

Tips
- Choose an audio‑capable chat model (examples):
  - STT_CHAT_MODEL=google/gemini-2.5-flash-lite
  - STT_CHAT_MODEL=openai/gpt-4o-audio-preview
- Tune segment size (seconds): SEG=60 (30–90 works well)
- Full control via script:
  - .venv/bin/python scripts/meeting_minutes.py --input Session.m4a --out minutes.txt --transcriber openrouter --stt-chat-model google/gemini-2.5-flash-lite

CLI highlights
- --transcriber: auto | openrouter | openai | local
  - auto prefers OpenRouter when OPENROUTER_API_KEY is set
- --stt-chat-model: OpenRouter chat model for audio (defaults to gemini‑2.5‑flash‑lite)
- --stt-model: OpenRouter STT endpoint model (if your account supports audio/transcriptions)
- --model: LLM for minutes (defaults auto: cheap for chunks, strong for final)

Alternate STT options
- OpenAI STT (keeps OpenRouter for minutes):
  - export OPENAI_API_KEY=...
  - make transcript FILE=Session.m4a ARGS="--transcriber openai --out minutes.txt"
- Local Whisper:
  - make install-whisper
  - make transcript FILE=Session.m4a ARGS="--transcriber local --out minutes.txt"

Notes
- The script auto‑splits long audio. It uses chat/completions with input_audio per OpenRouter docs.
- Optional headers supported: OPENROUTER_APP_TITLE, OPENROUTER_SITE_URL.
