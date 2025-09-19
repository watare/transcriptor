PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTHON := $(BIN)/python

.PHONY: venv install install-whisper run transcript clean help

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
# Usage: make transcript FILE=meeting.mp4 [ARGS="--out minutes.txt"]
transcript: venv
	@if [ -z "$(FILE)" ]; then echo "Usage: make transcript FILE=path/to.mp4 [ARGS=...]"; exit 2; fi
	$(PYTHON) scripts/meeting_minutes.py --input "$(FILE)" $(ARGS)

clean:
	rm -rf $(VENV)

help:
	@echo "Targets:"
	@echo "  make venv                 # create local .venv"
	@echo "  make install              # prepare base env"
	@echo "  make install-whisper      # install optional Whisper"
	@echo "  make run ARGS=...         # run script with args"
	@echo "  make transcript FILE=...  # run with --input file"
	@echo "  make clean                # remove .venv"
