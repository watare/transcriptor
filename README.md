# TalkScribe

A Python-based meeting minutes generator that uses OpenRouter API to transcribe audio/video files and generate structured meeting minutes with various levels of detail.

## Features

- **Multi-backend transcription**: OpenRouter (preferred), OpenAI, or local Whisper
- **Modular workflow**: Separate transcription and minutes generation
- **Multiple detail levels**: Basic, detailed, anonymous detailed, and ultra-detailed minutes
- **Smart model selection**: Cost-effective model selection for different phases
- **Context enhancement**: Optional web research integration
- **Chunked processing**: Handles large files efficiently

## Quick Start

### Setup
```bash
# 1. Create environment and install dependencies
make venv
make install

# 2. Configure API keys
cp .env.example .env
# Edit .env file and add your OpenRouter API key

# 3. Set environment variables (or use .env file)
export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"
```

### Basic Usage

#### Option 1: Two-step workflow (recommended)
```bash
# Step 1: Generate transcript
make transcript-only FILE=meeting.m4a OUT=transcript.txt

# Step 2: Generate minutes from transcript
make minutes-only TRANSCRIPT=transcript.txt OUT=minutes.txt
```

#### Option 2: Direct audio to minutes
```bash
make minutes FILE=meeting.m4a OUT=minutes.txt
```

## Detail Levels

### Basic Minutes (13/20 quality)
Simple summary with key points and decisions.
```bash
make minutes-only TRANSCRIPT=transcript.txt OUT=minutes_basic.txt
```

### Detailed Minutes (16/20 quality)
Includes technical details, specific numbers, company names, and concrete proposals.
```bash
make run ARGS="--transcript transcript.txt --minutes-only --out minutes_detailed.txt --synthesis map-reduce --chunk-tokens 4000 --model openai/gpt-4o --focus 'technical discussions, names, concrete proposals'"
```

### Anonymous Detailed Minutes (17/20 quality)
Detailed participant exchanges with anonymous Speaker A/B/C labels, disagreements, and discussion flow.
```bash
make run ARGS="--transcript transcript.txt --minutes-only --out minutes_anonymous.txt --synthesis map-reduce --chunk-tokens 6000 --model anthropic/claude-3.5-sonnet --focus 'participant exchanges using Speaker A/B/C, disagreements, discussion flow'"
```

### Ultra-Detailed Minutes (17+/20 quality)
Maximum detail with conversation flow, rebuttals, and specific exchanges.
```bash
make run ARGS="--transcript transcript.txt --minutes-only --out minutes_ultra.txt --synthesis map-reduce --chunk-tokens 7000 --model anthropic/claude-3.5-sonnet --focus 'detailed participant exchanges, who challenged whom, specific rebuttals, discussion flow'"
```

## Complete Workflow Examples

### Example 1: High-Quality Anonymous Minutes
```bash
# Step 1: Transcribe audio (120s segments for speed)
make transcript-only FILE=meeting.m4a OUT=transcript.txt SEG=120

# Step 2: Generate anonymous detailed minutes
make run ARGS="--transcript transcript.txt --minutes-only --out minutes.txt --synthesis map-reduce --chunk-tokens 6000 --model anthropic/claude-3.5-sonnet --focus 'participant exchanges using Speaker A/B/C, disagreements, discussion flow'"
```

### Example 2: Enhanced Minutes with Context
```bash
# Transcription
make transcript-only FILE=session.m4a OUT=transcript.txt

# Enhanced minutes with web research
make minutes-enhanced TRANSCRIPT=transcript.txt OUT=enhanced_minutes.txt
```

### Example 3: Multiple Formats
```bash
# One transcript, multiple minute formats
make transcript-only FILE=meeting.m4a OUT=transcript.txt

# Basic version
make minutes-only TRANSCRIPT=transcript.txt OUT=minutes_basic.txt

# Detailed version
make run ARGS="--transcript transcript.txt --minutes-only --out minutes_detailed.txt --synthesis map-reduce --chunk-tokens 4000 --model openai/gpt-4o --focus 'technical discussions, names, proposals'"

# Anonymous detailed version
make run ARGS="--transcript transcript.txt --minutes-only --out minutes_anonymous.txt --synthesis map-reduce --chunk-tokens 6000 --model anthropic/claude-3.5-sonnet --focus 'Speaker A/B/C exchanges, disagreements, flow'"
```

## Available Commands

| Command | Purpose | Quality Level |
|---------|---------|---------------|
| `make transcript-only` | Audio → transcript only | - |
| `make minutes-only` | Basic minutes from transcript | 13/20 |
| `make minutes-enhanced` | Context-enriched minutes | 15/20 |
| `make minutes` | Direct audio → minutes | 13/20 |
| `make run ARGS="..."` | Custom parameters | Variable |

## Parameters

### Key Options
- `--synthesis`: `auto`, `map-reduce`, `full`
- `--chunk-tokens`: Size of chunks for processing (2000-7000)
- `--model`: LLM model (`openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, etc.)
- `--focus`: Specific instructions for content focus
- `--enhance`: Add web research context
- `--anonymize`: Use Speaker A/B/C labels

### Model Recommendations
- **GPT-4o**: Good for technical details and structured output
- **Claude 3.5 Sonnet**: Excellent for conversation flow and nuanced discussions
- **Gemini models**: Alternative options

## Environment Variables

Required:
- `OPENROUTER_API_KEY`: Your OpenRouter API key

Optional:
- `OPENAI_API_KEY`: For OpenAI fallback
- `STT_CHAT_SEGMENT_SECONDS`: Audio segment duration (default: 60)
- `OPENROUTER_STT_CHAT_MODEL`: Default chat model for transcription

## Tips for Best Results

1. **Use the two-step workflow** for reliability and efficiency
2. **Adjust chunk sizes**: Larger chunks (6000-7000) preserve conversation context
3. **Choose the right model**: Claude for nuanced discussions, GPT-4o for technical precision
4. **Use specific focus instructions** to guide the AI toward your desired content
5. **For long meetings**: Use map-reduce synthesis to avoid context limits

## File Structure
```
transcriptor/
├── scripts/meeting_minutes.py  # Main application
├── Makefile                   # Build automation
├── README.md                  # This file
├── CLAUDE.md                  # Project instructions
└── .venv/                     # Virtual environment (created by make venv)
```

## Troubleshooting

- **OpenRouter HTML errors**: Try map-reduce synthesis with smaller chunks
- **Context limit exceeded**: Reduce chunk-tokens or use map-reduce mode
- **Empty transcript**: Check API keys and audio file format
- **Poor quality minutes**: Increase chunk-tokens and use more specific focus instructions

## Help

Run `make help` for quick command reference or check the examples above.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see the LICENSE file for details.

## Repository

[https://github.com/watare/talkscribe](https://github.com/watare/talkscribe)
