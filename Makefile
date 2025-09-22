PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTHON := $(BIN)/python

.PHONY: venv install install-whisper run transcript minutes clean help

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip

# Base install (no mandatory deps required for summarization)
install: venv
	@echo "Base environment ready. Install extras with 'make install-whisper' if you need local transcription."

# Optional: local transcription via Whisper (requires ffmpeg installed on your system)
install-whisper: venv
	$(PIP) install -U openai-whisper

# Run the meeting minutes generator inside the venv
# Example: make run ARGS="--input meeting.mp4 --out minutes.txt --model anthropic/claude-3.5-sonnet"
run: venv
	$(PYTHON) scripts/meeting_minutes.py $(ARGS)

# Convenience: transcribe + summarize from a file
# Usage:
#   make transcript FILE=meeting.mp4 [ARGS="--out minutes.txt"]
#   make transcript ARGS="--file meeting.mp4 --out minutes.txt"
transcript: venv
	@# Accept either FILE=... or ARGS containing --input/--file
	@if [ -z "$(FILE)" ] && ! printf "%s" "$(ARGS)" | grep -Eq -- '--(input|file)(=|[[:space:]])'; then \
		echo "Usage: make transcript FILE=path/to.mp4 [ARGS=...]"; \
		echo "   or: make transcript ARGS=\"--file path/to.mp4 [other args]\""; \
		echo "   or: make transcript FILE=path/to.mp4 OUT=minutes.txt"; \
		exit 2; \
	fi
	$(PYTHON) scripts/meeting_minutes.py $(if $(FILE),--input "$(FILE)",) $(if $(OUT),--out "$(OUT)",) $(ARGS)

# Convenience: OpenRouter-only transcription + minutes (audio-capable chat models)
# Usage examples:
#   make minutes FILE=Session.m4a OUT=minutes.txt
#   make minutes FILE=Session.m4a OUT=minutes.txt STT_CHAT_MODEL=openai/gpt-4o-audio-preview SEG=60
# Overrides:
#   STT_CHAT_MODEL (default: google/gemini-2.5-flash-lite)
#   SEG (segment seconds for chat STT; default: 60)
STT_CHAT_MODEL ?= google/gemini-2.5-flash-lite
SEG ?= 60
minutes: venv
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make minutes FILE=path/to/audio_or_video [OUT=minutes.txt] [STT_CHAT_MODEL=...] [SEG=60]"; \
		exit 2; \
	fi
	STT_CHAT_SEGMENT_SECONDS=$(SEG) $(PYTHON) scripts/meeting_minutes.py --transcriber openrouter --stt-chat-model $(STT_CHAT_MODEL) --input "$(FILE)" $(if $(OUT),--out "$(OUT)",)

clean:
	rm -rf $(VENV)

help:
	@echo "Targets:"
	@echo "  make venv                 # create local .venv"
	@echo "  make install              # prepare base env"
	@echo "  make install-whisper      # install optional Whisper"
	@echo "  make run ARGS=...         # run script with args"
	@echo "  make transcript FILE=...  # run with --input file"
	@echo "  make transcript ARGS=\"--file path\"  # alternative"
	@echo "  make transcript FILE=meeting.mp4 OUT=minutes.txt  # simpler"
	@echo "  e.g.: make transcript FILE=meeting.mp4 ARGS=\"--transcriber local\""
	@echo "  make minutes FILE=Session.m4a OUT=minutes.txt  # OpenRouter chat STT + minutes"
	@echo "  make clean                # remove .venv"
