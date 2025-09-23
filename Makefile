PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTHON := $(BIN)/python

.PHONY: venv install install-whisper run transcript minutes transcript-only minutes-only minutes-enhanced clean help

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

# Transcript only: just transcribe audio/video, no minutes generation
# Usage: make transcript-only FILE=Session.m4a OUT=transcript.txt
transcript-only: venv
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make transcript-only FILE=path/to/audio_or_video OUT=transcript.txt [STT_CHAT_MODEL=...] [SEG=120]"; \
		exit 2; \
	fi
	STT_CHAT_SEGMENT_SECONDS=$(SEG) $(PYTHON) scripts/meeting_minutes.py --transcriber openrouter --stt-chat-model $(STT_CHAT_MODEL) --input "$(FILE)" --transcript-only $(if $(OUT),--out "$(OUT)",)

# Minutes only: generate minutes from existing transcript
# Usage: make minutes-only TRANSCRIPT=transcript.txt OUT=minutes.txt
minutes-only: venv
	@if [ -z "$(TRANSCRIPT)" ]; then \
		echo "Usage: make minutes-only TRANSCRIPT=path/to/transcript.txt OUT=minutes.txt [MODEL=...]"; \
		exit 2; \
	fi
	$(PYTHON) scripts/meeting_minutes.py --transcript "$(TRANSCRIPT)" --minutes-only $(if $(OUT),--out "$(OUT)",) $(if $(MODEL),--model "$(MODEL)",)

# Enhanced minutes: generate contextually enriched minutes with web research
# Usage: make minutes-enhanced TRANSCRIPT=transcript.txt OUT=enhanced_minutes.txt
minutes-enhanced: venv
	@if [ -z "$(TRANSCRIPT)" ]; then \
		echo "Usage: make minutes-enhanced TRANSCRIPT=path/to/transcript.txt OUT=enhanced_minutes.txt [MODEL=...]"; \
		exit 2; \
	fi
	$(PYTHON) scripts/meeting_minutes.py --transcript "$(TRANSCRIPT)" --minutes-only --enhance --synthesis map-reduce --chunk-tokens 3000 $(if $(OUT),--out "$(OUT)",) $(if $(MODEL),--model "$(MODEL)",)

clean:
	rm -rf $(VENV)

help:
	@echo "Meeting Minutes Generator - Usage Examples:"
	@echo ""
	@echo "SETUP:"
	@echo "  make venv                 # create local .venv"
	@echo "  make install              # prepare base env"
	@echo "  make install-whisper      # install optional Whisper"
	@echo ""
	@echo "WORKFLOW EXAMPLES:"
	@echo ""
	@echo "1. FULL WORKFLOW (audio → transcript → minutes):"
	@echo "   make transcript-only FILE=meeting.m4a OUT=transcript.txt"
	@echo "   make minutes-only TRANSCRIPT=transcript.txt OUT=minutes.txt"
	@echo ""
	@echo "2. DIFFERENT DETAIL LEVELS:"
	@echo ""
	@echo "   Basic minutes (13/20):"
	@echo "   make minutes-only TRANSCRIPT=transcript.txt OUT=minutes_basic.txt"
	@echo ""
	@echo "   Detailed minutes (16/20):"
	@echo "   make run ARGS=\"--transcript transcript.txt --minutes-only --out minutes_detailed.txt --synthesis map-reduce --chunk-tokens 4000 --model openai/gpt-4o --focus 'technical discussions, names, concrete proposals'\""
	@echo ""
	@echo "   Anonymous detailed (17/20):"
	@echo "   make run ARGS=\"--transcript transcript.txt --minutes-only --out minutes_anon.txt --synthesis map-reduce --chunk-tokens 6000 --model anthropic/claude-3.5-sonnet --focus 'participant exchanges using Speaker A/B/C, disagreements, discussion flow'\""
	@echo ""
	@echo "ALL COMMANDS:"
	@echo "  make transcript-only FILE=audio.m4a OUT=transcript.txt  # transcription only"
	@echo "  make minutes-only TRANSCRIPT=transcript.txt OUT=minutes.txt  # basic minutes"
	@echo "  make minutes-enhanced TRANSCRIPT=transcript.txt OUT=enhanced.txt  # with context research"
	@echo "  make minutes FILE=audio.m4a OUT=minutes.txt  # direct audio → minutes"
	@echo "  make run ARGS=\"...\"        # custom arguments"
	@echo "  make clean                # remove .venv"
