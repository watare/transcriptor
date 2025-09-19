#!/usr/bin/env python3
"""
Meeting Minutes Generator from MP4 using OpenRouter

Features
- Optional local transcription via Whisper if installed; or pass an existing transcript.
- Chunk-aware summarization to handle long meetings.
- Produces email-ready minutes with subject and structured sections.

Usage
  python scripts/meeting_minutes.py --input path/to/video.mp4 --out minutes.txt \
    --model anthropic/claude-3.5-sonnet

  # If you already have a transcript
  python scripts/meeting_minutes.py --transcript path/to/transcript.txt --out minutes.txt

Environment
- Set OPENROUTER_API_KEY with your key from https://openrouter.ai/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import urllib.request
    import urllib.error
except Exception as e:  # pragma: no cover
    print(f"Error importing urllib: {e}", file=sys.stderr)
    sys.exit(1)


DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def try_import_whisper():
    try:
        import whisper  # type: ignore

        return whisper
    except Exception:
        return None


def transcribe_with_whisper(input_path: Path, model_size: str = "small") -> str:
    whisper = try_import_whisper()
    if whisper is None:
        raise RuntimeError(
            "Whisper is not installed. Install with `pip install openai-whisper` "
            "and ensure ffmpeg is available, or provide --transcript instead."
        )
    eprint(f"Transcribing with Whisper ({model_size})… this can take a while")
    model = whisper.load_model(model_size)
    # Disable fp16 to work on CPU-only machines
    result = model.transcribe(str(input_path), fp16=False)
    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError("Transcription returned empty text.")
    return text


def estimate_tokens(chars: int) -> int:
    # Naive approximation: ~4 chars per token
    return max(1, math.ceil(chars / 4))


def chunk_text(text: str, target_tokens: int = 3000) -> List[str]:
    # Split by paragraphs/sentences to keep semantics
    separators = ["\n\n", "\n", ". "]
    units = [text]
    for sep in separators:
        if len(units) == 1:
            units = text.split(sep)
        else:
            break
    # If still one big unit, fall back to fixed-size chunks
    if len(units) == 1 and estimate_tokens(len(text)) > target_tokens:
        chunk_chars = target_tokens * 4
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    chunks: List[str] = []
    cur = []
    cur_len = 0
    max_chars = target_tokens * 4
    for u in units:
        u = u.strip()
        if not u:
            continue
        if cur_len + len(u) + 1 <= max_chars:
            cur.append(u)
            cur_len += len(u) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            cur = [u]
            cur_len = len(u)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def call_openrouter(
    api_key: str,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            parsed = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTPError {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter URLError: {e}")

    try:
        content = parsed["choices"][0]["message"]["content"]
        if isinstance(content, list):
            # Some providers may return array of parts
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {e}; got: {parsed}")


def build_chunk_prompt(language: str) -> List[Dict[str, str]]:
    system = (
        "You are an expert meeting minute-taker. Extract precise, concise notes "
        "from the transcript chunk. Detect and preserve context (project, domain, "
        "customer, product names), keep timestamps if present, list decisions and "
        "action items with owners and tentative due dates if implied. Output in "
        f"{language} using bullet points."
    )
    return [{"role": "system", "content": system}]


def build_final_prompt(language: str) -> List[Dict[str, str]]:
    system = (
        "You are a senior project manager creating email-ready meeting minutes. "
        "Synthesize chunked notes into a cohesive document. Infer context "
        "(team, customer, project, goals) from cues. Be factual and concise. "
        "Include: Subject, Context, Attendees (if inferable), Agenda (if inferable), "
        "Key Discussion Points, Decisions, Action Items (owner, due, priority), "
        "Risks/Blockers, Next Steps, and an Executive Summary. Use clear, "
        f"email-friendly formatting in {language}."
    )
    return [{"role": "system", "content": system}]


def summarize_chunks(
    api_key: str,
    model: str,
    chunks: List[str],
    language: str,
    temperature: float,
) -> List[str]:
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        eprint(f"Summarizing chunk {i}/{len(chunks)}…")
        messages = build_chunk_prompt(language) + [
            {
                "role": "user",
                "content": (
                    "Transcript chunk: "
                    + chunk
                    + "\n\nReturn structured bullet notes only."
                ),
            }
        ]
        text = call_openrouter(api_key, messages, model=model, temperature=temperature)
        summaries.append(text)
        # Gentle pacing to avoid rate limits
        time.sleep(0.5)
    return summaries


def synthesize_minutes(
    api_key: str,
    model: str,
    chunk_summaries: List[str],
    language: str,
    subject_prefix: Optional[str],
    temperature: float,
) -> str:
    messages = build_final_prompt(language)
    content = (
        (f"Subject prefix: {subject_prefix}\n" if subject_prefix else "")
        + "Here are the chunked notes to synthesize into final minutes:\n\n"
        + "\n\n---\n\n".join(chunk_summaries)
        + "\n\nConstraints: keep it concise, email-ready, and action-oriented."
    )
    messages.append({"role": "user", "content": content})
    return call_openrouter(api_key, messages, model=model, temperature=temperature)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate meeting minutes from MP4 using OpenRouter")
    g_in = parser.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--input", "--file", dest="input", type=str, help="Path to input MP4 (or audio)")
    g_in.add_argument("--transcript", type=str, help="Path to existing transcript .txt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (overrides OPENROUTER_API_KEY)")
    parser.add_argument("--language", type=str, default="English", help="Output language (e.g., English, Français)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--subject-prefix", type=str, default="Meeting Minutes", help="Subject line prefix")
    parser.add_argument("--out", type=str, default="-", help="Output file or '-' for stdout")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model size if used (tiny|base|small|medium|large)")
    parser.add_argument("--chunk-tokens", type=int, default=3000, help="Approx tokens per chunk for map-reduce")

    args = parser.parse_args(argv)

    # Load .env if present (no external dependency)
    def load_dotenv(paths: List[str]) -> None:
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            try:
                for raw in path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            except Exception:
                # Non-fatal if .env parsing fails
                pass

    load_dotenv([".env", ".env.local"])  # does not override existing env vars

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        eprint("Missing OPENROUTER_API_KEY. Export it before running.")
        return 2

    transcript_text = ""
    if args.transcript:
        tpath = Path(args.transcript)
        if not tpath.exists():
            eprint(f"Transcript not found: {tpath}")
            return 2
        transcript_text = read_text(tpath).strip()
    else:
        ipath = Path(args.input)
        if not ipath.exists():
            eprint(f"Input not found: {ipath}")
            return 2
        try:
            transcript_text = transcribe_with_whisper(ipath, model_size=args.whisper_model)
        except Exception as e:
            eprint(str(e))
            eprint("Tip: provide --transcript if you prefer to skip local transcription.")
            return 2

    if not transcript_text:
        eprint("Empty transcript; aborting.")
        return 2

    # Prepare chunks if needed
    tokens = estimate_tokens(len(transcript_text))
    target = max(500, int(args.chunk_tokens))
    if tokens > target * 1.2:
        chunks = chunk_text(transcript_text, target_tokens=target)
    else:
        chunks = [transcript_text]

    eprint(f"Transcript length ~{tokens} tokens → {len(chunks)} chunk(s)")

    try:
        summaries = summarize_chunks(
            api_key=api_key,
            model=args.model,
            chunks=chunks,
            language=args.language,
            temperature=args.temperature,
        )
        final_minutes = synthesize_minutes(
            api_key=api_key,
            model=args.model,
            chunk_summaries=summaries,
            language=args.language,
            subject_prefix=args.subject_prefix,
            temperature=args.temperature,
        )
    except Exception as e:
        eprint(f"OpenRouter error: {e}")
        return 3

    if args.out == "-":
        print(final_minutes)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_minutes, encoding="utf-8")
        eprint(f"Wrote minutes to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
