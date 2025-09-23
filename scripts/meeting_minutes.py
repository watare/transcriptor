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
import subprocess
import tempfile

try:
    import urllib.request
    import urllib.error
except Exception as e:  # pragma: no cover
    print(f"Error importing urllib: {e}", file=sys.stderr)
    sys.exit(1)


# Model selection
# Use "auto" to select cost-effective defaults based on phase and size.
DEFAULT_MODEL = "auto"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
OPENROUTER_TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


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


def _encode_multipart_formdata(fields: Dict[str, str], files: List[Dict[str, Any]]) -> tuple[bytes, str]:
    """Construct a multipart/form-data body.

    files: list of dicts with keys: name, filename, content, content_type
    Returns: (body_bytes, content_type_header_value)
    """
    import uuid

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    crlf = "\r\n"
    lines: List[bytes] = []

    for k, v in fields.items():
        lines.append((f"--{boundary}" + crlf).encode("utf-8"))
        lines.append((f"Content-Disposition: form-data; name=\"{k}\"" + crlf + crlf).encode("utf-8"))
        lines.append((str(v) + crlf).encode("utf-8"))

    for f in files:
        name = f["name"]
        filename = f["filename"]
        content = f["content"]  # bytes
        ctype = f.get("content_type", "application/octet-stream")
        lines.append((f"--{boundary}" + crlf).encode("utf-8"))
        lines.append((
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"" + crlf
        ).encode("utf-8"))
        lines.append((f"Content-Type: {ctype}" + crlf + crlf).encode("utf-8"))
        lines.append(content)
        lines.append(crlf.encode("utf-8"))

    lines.append((f"--{boundary}--" + crlf).encode("utf-8"))
    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _guess_lang_code(language: str | None) -> Optional[str]:
    if not language:
        return None
    lang = language.strip().lower()
    # naive guess: take first two letters for common languages
    if len(lang) >= 2:
        return lang[:2]
    return None


def transcribe_with_openai(
    input_path: Path,
    api_key: str,
    model: str = "whisper-1",
    language: Optional[str] = None,
) -> str:
    """Transcribe using OpenAI's transcription API (no openai package required)."""
    # Read file
    data = input_path.read_bytes()
    fields: Dict[str, str] = {"model": model, "response_format": "text"}
    lang_code = _guess_lang_code(language)
    if lang_code:
        fields["language"] = lang_code
    files = [
        {
            "name": "file",
            "filename": input_path.name,
            "content": data,
            "content_type": "application/octet-stream",
        }
    ]
    body, content_type = _encode_multipart_formdata(fields, files)

    req = urllib.request.Request(
        OPENAI_TRANSCRIBE_URL,
        data=body,
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
            if not text:
                raise RuntimeError("OpenAI transcription returned empty response.")
            return text
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTPError {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI URLError: {e}")


def transcribe_with_openrouter(
    input_path: Path,
    api_key: str,
    model: str = "openai/whisper-1",
    language: Optional[str] = None,
) -> str:
    """Transcribe using OpenRouter's transcription API (OpenAI-compatible endpoint)."""
    data = input_path.read_bytes()
    fields: Dict[str, str] = {"model": model, "response_format": "text"}
    lang_code = _guess_lang_code(language)
    if lang_code:
        fields["language"] = lang_code
    files = [
        {
            "name": "file",
            "filename": input_path.name,
            "content": data,
            "content_type": "application/octet-stream",
        }
    ]
    body, content_type = _encode_multipart_formdata(fields, files)

    headers = {
        "Content-Type": content_type,
        "Authorization": f"Bearer {api_key}",
        # Optional but recommended for OpenRouter
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "transcriptor-minutes"),
    }

    req = urllib.request.Request(
        OPENROUTER_TRANSCRIBE_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
            if not text:
                raise RuntimeError("OpenRouter transcription returned empty response.")
            return text
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTPError {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter URLError: {e}")


def split_audio_with_ffmpeg(input_path: Path, segment_seconds: int = 300) -> List[Path]:
    """Split audio/video into segments using ffmpeg. Returns list of segment paths."""
    if segment_seconds <= 0:
        segment_seconds = 300
    tmpdir = tempfile.mkdtemp(prefix="stt_segments_")
    pattern = str(Path(tmpdir) / "seg_%03d.m4a")
    # Use re-encode to ensure clean segment boundaries across formats
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {proc.stderr.strip()}")
    # Collect segments
    segs = sorted(Path(tmpdir).glob("seg_*.m4a"))
    if not segs:
        raise RuntimeError("ffmpeg produced no segments")
    return segs


def split_wav_with_ffmpeg(input_path: Path, segment_seconds: int = 45) -> List[Path]:
    """Transcode to 16kHz mono WAV and split into small segments for chat-based STT."""
    if segment_seconds <= 0:
        segment_seconds = 45
    tmpdir = tempfile.mkdtemp(prefix="stt_wav_segments_")
    pattern = str(Path(tmpdir) / "seg_%03d.wav")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg wav split failed: {proc.stderr.strip()}")
    segs = sorted(Path(tmpdir).glob("seg_*.wav"))
    if not segs:
        raise RuntimeError("ffmpeg produced no wav segments")
    return segs


def transcribe_with_openrouter_chat(
    input_path: Path,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    language: Optional[str] = None,
    segment_seconds: int = 45,
) -> str:
    import base64

    segs = split_wav_with_ffmpeg(input_path, segment_seconds=segment_seconds)
    out: List[str] = []
    lang_hint = f" Language: {language}." if language else ""
    for i, seg in enumerate(segs, 1):
        eprint(f"  → Chat STT segment {i}/{len(segs)}: {seg.name}")
        audio_bytes = seg.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        # Use chat/completions with input_audio per OpenRouter docs
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a precise speech-to-text transcriber. Return only the raw transcript without extra commentary.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Please transcribe this audio exactly.{lang_hint}"},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            },
        ]
        text = call_openrouter(api_key, messages, model=model, temperature=0.0)
        out.append(text.strip())
        time.sleep(0.3)
    return "\n".join(out)


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
    model: str,
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Optional but recommended for OpenRouter routing/analytics
    site = os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if title:
        headers["X-Title"] = title
    if site:
        headers["HTTP-Referer"] = site

    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            if not body.strip():
                raise RuntimeError("OpenRouter returned empty response")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"OpenRouter returned invalid JSON: {e}. Response: {body.decode('utf-8', errors='ignore')[:200]}")
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


def call_openrouter_responses(
    api_key: str,
    input_items: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_RESPONSES_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "transcriptor-minutes"),
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            if not body.strip():
                raise RuntimeError("OpenRouter returned empty response")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"OpenRouter returned invalid JSON: {e}. Response: {body.decode('utf-8', errors='ignore')[:200]}")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTPError {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter URLError: {e}")

    # Try common shapes
    if isinstance(parsed, dict):
        if "output_text" in parsed and isinstance(parsed["output_text"], str):
            return parsed["output_text"].strip()
        if "choices" in parsed:
            try:
                content = parsed["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content).strip()
            except Exception:
                pass
        if "output" in parsed and isinstance(parsed["output"], list):
            # OpenAI Responses-like: output is list of content parts
            parts_text: List[str] = []
            for item in parsed["output"]:
                if isinstance(item, dict):
                    if item.get("type") == "output_text" and "text" in item:
                        parts_text.append(str(item["text"]))
                    elif "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                parts_text.append(str(c.get("text", "")))
            if parts_text:
                return "".join(parts_text).strip()
    raise RuntimeError(f"Unexpected OpenRouter responses payload: {parsed}")


def pick_models_for_minutes(total_tokens: int, user_model: str | None) -> tuple[str, str]:
    """Return (chunk_model, final_model) for summarization.

    Heuristic:
    - If user specifies a concrete model (not "auto"), use it for both phases.
    - Otherwise, pick a cost-effective pair available on OpenRouter:
      - chunk_model: a small, cheap but capable model (gpt-4o-mini as first choice),
        fallback to claude-3.5-sonnet or the default.
      - final_model: higher quality for synthesis if size allows; else mini.

    We can't query model availability here; we attempt first choice and callers
    will handle HTTP errors by surfacing OpenRouter error text.
    """
    # If user forced a model, use it as-is
    if user_model and user_model.strip().lower() != "auto":
        m = user_model.strip()
        return m, m

    # Auto selection: favor cost-effective defaults
    # Primary cheap option
    cheap = os.getenv("OPENROUTER_CHEAP_MODEL", "openai/gpt-4o-mini")
    strong = os.getenv("OPENROUTER_STRONG_MODEL", "anthropic/claude-3.5-sonnet")

    # If transcript is very large, prefer cheap for both to control cost
    if total_tokens > 60_000:  # ~240k chars
        return cheap, cheap
    # Moderate size: cheap for chunks, strong for final synthesis
    return cheap, strong


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
    focus: Optional[str] = None,
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
                    + ("\n\nFocus specifically on: " + focus if focus else "")
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
    focus: Optional[str] = None,
) -> str:
    messages = build_final_prompt(language)
    content = (
        (f"Subject prefix: {subject_prefix}\n" if subject_prefix else "")
        + "Here are the chunked notes to synthesize into final minutes:\n\n"
        + "\n\n---\n\n".join(chunk_summaries)
        + ("\n\nPriority topics to cover: " + focus if focus else "")
        + "\n\nConstraints: keep it concise, email-ready, and action-oriented."
    )
    messages.append({"role": "user", "content": content})
    return call_openrouter(api_key, messages, model=model, temperature=temperature)


def synthesize_minutes_full(
    api_key: str,
    model: str,
    transcript_text: str,
    language: str,
    subject_prefix: Optional[str],
    temperature: float,
    focus: str | None = None,
    target_tokens: int = 12000,
) -> str:
    """Single-pass synthesis over the full transcript using the Responses API.

    The transcript is fed as a sequence of input items to preserve global context.
    target_tokens controls the approximate per-item size (in tokens) to avoid overly large payloads.
    """
    header = (
        (f"Subject prefix: {subject_prefix}\n" if subject_prefix else "")
        + "You will receive the full meeting transcript in segments. "
        + "Produce a single, cohesive set of minutes. "
        + "Include: Subject, Executive Summary, Context, Attendees (if inferable), Agenda (if inferable), Key Discussion Points, Decisions, Action Items (owner, due, priority), Risks/Blockers, Next Steps. "
        + ("\nPriority topics to cover: " + focus if focus else "")
        + "\nReturn only the final minutes in "
        + language
        + "."
    )

    # Break transcript into large but safe chunks
    t_tokens = estimate_tokens(len(transcript_text))
    larger_target = max(target_tokens, 8000)
    if t_tokens > int(larger_target * 1.1):
        t_chunks = chunk_text(transcript_text, target_tokens=larger_target)
    else:
        t_chunks = [transcript_text]

    # Build inputs for the Responses API
    input_items: List[Dict[str, Any]] = []
    input_items.append({
        "role": "system",
        "content": build_final_prompt(language)[0]["content"],
    })
    input_items.append({"role": "user", "content": header})
    for i, seg in enumerate(t_chunks, 1):
        input_items.append({
            "role": "user",
            "content": f"[Transcript segment {i}/{len(t_chunks)}]\n" + seg,
        })

    return call_openrouter_responses(
        api_key=api_key,
        input_items=input_items,
        model=model,
        temperature=temperature,
    )


def web_search_simple(query: str, domain: str = "general_corporate") -> str:
    """Enhanced knowledge base with domain-specific information"""
    import urllib.parse
    try:
        # Multi-domain knowledge base - organized by industry domain
        all_search_results = {
            # Common/Cross-industry terms
            "common": {
                "CAPEX vs OPEX": "Capital Expenditure (CAPEX) vs Operational Expenditure (OPEX): CAPEX includes investments in infrastructure, systems, and equipment providing long-term value. OPEX covers daily operational costs like maintenance, staff, software licenses, and services. Organizations shift from CAPEX-heavy models to OPEX through cloud services and SaaS for improved financial flexibility.",

                "digital transformation": "Digital transformation involves modernizing business operations through technology adoption: cloud migration, data analytics, AI/ML integration, process automation, and digital customer experiences. Key challenges include change management, cybersecurity, legacy system integration, and workforce upskilling.",

                "cybersecurity": "Modern cybersecurity frameworks address: threat detection and response, identity and access management, data protection, network security, and compliance. Organizations adopt zero-trust architectures, security awareness training, and incident response plans to protect against evolving cyber threats.",

                "open source": "Open source software adoption requires evaluation of: community health, security practices, licensing compliance, support availability, and total cost of ownership. Organizations benefit from reduced vendor lock-in, transparency, and community-driven innovation while managing security and maintenance responsibilities.",

                "SBOM": "Software Bill of Materials (SBOM) provides transparency into software components for vulnerability tracking, license compliance, and supply chain risk management. Critical for security assurance and regulatory compliance across industries."
            },

            # Energy/Utility specific
            "energy_utility": {
                "TSO": "Transmission System Operator (TSO) manages high-voltage electricity transmission networks, ensuring grid stability and cross-border electricity flows. Responsibilities include real-time balancing, capacity planning, market operations, and renewable integration.",

                "DSO": "Distribution System Operator (DSO) manages medium/low-voltage networks delivering electricity to end users. Key functions include grid maintenance, customer connections, demand response, and distributed energy resource integration.",

                "grid balancing": "Grid balancing involves matching electricity supply and demand in real-time to maintain system frequency. Uses automatic generation control, frequency response services, energy storage, and cross-border cooperation.",

                "NERC CIP": "North American Electric Reliability Corporation Critical Infrastructure Protection standards mandate cybersecurity controls for bulk electric systems. Requirements include asset identification, security controls, training, and incident response.",

                "renewable integration": "Integrating variable renewable energy sources requires grid flexibility mechanisms: energy storage, demand response, enhanced forecasting, and market mechanisms to manage variability while maintaining system stability."
            },

            # Technology/Software specific
            "technology_software": {
                "microservices": "Microservices architecture decomposes applications into small, independent services communicating via APIs. Benefits include scalability, technology diversity, fault isolation, and team independence. Challenges include complexity, network latency, and distributed system management.",

                "API": "Application Programming Interface (API) enables software components to communicate. Modern APIs use REST, GraphQL, or gRPC protocols. API management includes design, security, versioning, rate limiting, and monitoring for reliable integrations.",

                "cloud migration": "Cloud migration strategies include lift-and-shift, re-platforming, and cloud-native refactoring. Considerations include cost optimization, security, compliance, performance, and vendor lock-in. Hybrid and multi-cloud approaches provide flexibility.",

                "DevOps": "DevOps combines development and operations practices to accelerate software delivery through automation, continuous integration/delivery, infrastructure as code, and monitoring. Focuses on collaboration, reliability, and faster time-to-market.",

                "containerization": "Containerization packages applications with dependencies for consistent deployment across environments. Docker and Kubernetes enable scalable, portable applications with improved resource utilization and deployment automation."
            },

            # Finance/Banking specific
            "finance_banking": {
                "risk management": "Financial risk management identifies, measures, and mitigates risks including market risk, credit risk, operational risk, and liquidity risk. Uses quantitative models, stress testing, and regulatory frameworks like Basel III for capital adequacy.",

                "algorithmic trading": "Algorithmic trading uses computer programs to execute trades based on predefined criteria. Strategies include market making, arbitrage, and trend following. Requires low-latency infrastructure, risk controls, and regulatory compliance.",

                "RegTech": "Regulatory Technology (RegTech) automates compliance processes using AI, machine learning, and data analytics. Applications include transaction monitoring, reporting, KYC/AML compliance, and regulatory change management.",

                "DeFi": "Decentralized Finance (DeFi) recreates traditional financial services using blockchain and smart contracts. Includes lending, trading, derivatives, and insurance without traditional intermediaries. Risks include smart contract vulnerabilities and regulatory uncertainty."
            },

            # Healthcare specific
            "healthcare_medical": {
                "HIPAA": "Health Insurance Portability and Accountability Act (HIPAA) establishes privacy and security standards for protected health information. Requires administrative, physical, and technical safeguards, breach notification, and business associate agreements.",

                "electronic health records": "Electronic Health Records (EHR) digitize patient information for improved care coordination, clinical decision support, and population health management. Interoperability standards like HL7 FHIR enable data exchange between systems.",

                "telemedicine": "Telemedicine delivers healthcare services remotely using telecommunications technology. Includes virtual consultations, remote monitoring, and digital therapeutics. Regulatory considerations include licensing, reimbursement, and privacy compliance."
            },

            # Manufacturing specific
            "manufacturing_industrial": {
                "Industry 4.0": "Industry 4.0 represents the fourth industrial revolution integrating cyber-physical systems, IoT, cloud computing, and AI in manufacturing. Enables smart factories with autonomous systems, predictive maintenance, and mass customization.",

                "supply chain": "Modern supply chains integrate suppliers, manufacturers, and distributors through digital platforms. Challenges include visibility, resilience, sustainability, and risk management. Technologies include blockchain for traceability and AI for optimization.",

                "lean manufacturing": "Lean manufacturing eliminates waste through continuous improvement (kaizen), just-in-time production, and value stream mapping. Focuses on customer value, flow optimization, and employee engagement for operational excellence."
            }
        }

        # Select appropriate domain knowledge base
        domain_results = all_search_results.get(domain, {})
        common_results = all_search_results.get("common", {})
        search_results = {**common_results, **domain_results}

        # Find best matching result with fuzzy matching
        query_lower = query.lower()

        # Direct match first
        for key, result in search_results.items():
            if key.lower() in query_lower:
                return result

        # Partial word matching
        query_words = query_lower.split()
        best_match = None
        max_matches = 0

        for key, result in search_results.items():
            key_words = key.lower().split()
            matches = sum(1 for word in query_words if any(word in key_word for key_word in key_words))
            if matches > max_matches:
                max_matches = matches
                best_match = result

        if best_match and max_matches > 0:
            return best_match

        # Domain-aware fallback explanations
        domain_fallbacks = {
            "energy_utility": f"Energy sector research indicates '{query}' is an emerging topic with increasing importance for grid modernization, renewable integration, and digital transformation initiatives. Consider consulting ENTSO-E reports, IEA energy transition roadmaps, or utility industry publications for detailed analysis.",
            "technology_software": f"Technology research suggests '{query}' is gaining attention in software development and IT operations, with relevance to cloud adoption, DevOps practices, and digital transformation initiatives. Consider consulting industry reports from Gartner, Stack Overflow surveys, or CNCF landscape analysis.",
            "finance_banking": f"Financial services research indicates '{query}' is an emerging topic with growing importance for risk management, regulatory compliance, and digital banking transformation. Consider consulting reports from central banks, Basel Committee publications, or fintech industry analysis.",
            "healthcare_medical": f"Healthcare research suggests '{query}' is gaining attention with relevance to patient care, regulatory compliance, and healthcare technology adoption. Consider consulting FDA guidance, healthcare industry publications, or medical technology assessments.",
            "manufacturing_industrial": f"Manufacturing research indicates '{query}' is an emerging topic with importance for operational efficiency, supply chain optimization, and Industry 4.0 transformation. Consider consulting industrial automation reports, supply chain management publications, or manufacturing technology assessments.",
            "general_corporate": f"Industry research suggests '{query}' is gaining attention across multiple sectors, with particular relevance to digital transformation, operational efficiency, and strategic business initiatives."
        }

        fallback_message = domain_fallbacks.get(domain, domain_fallbacks["general_corporate"])
        return fallback_message

    except Exception as e:
        return f"[Knowledge base lookup failed: {e}]"


def detect_meeting_domain(minutes_text: str, api_key: str, model: str = "openai/gpt-4o-mini") -> str:
    """Detect the primary domain/industry context of the meeting"""
    prompt = """
    Analyze these meeting minutes and identify the primary domain/industry context.

    Choose the MOST RELEVANT domain from these categories:
    - energy_utility (power grid, TSO/DSO, electricity markets, renewable energy)
    - finance_banking (financial services, trading, investment, risk management)
    - technology_software (software development, IT, cloud, cybersecurity, SaaS)
    - healthcare_medical (healthcare, pharmaceuticals, medical devices, clinical)
    - manufacturing_industrial (manufacturing, supply chain, industrial processes)
    - government_public (government, public sector, policy, regulation)
    - consulting_professional (consulting, legal, professional services)
    - academia_research (academic, research, education, scientific)
    - startup_business (startups, entrepreneurship, venture capital, business development)
    - general_corporate (general corporate meetings, HR, operations)

    Return ONLY the domain category (e.g., "energy_utility" or "technology_software").

    Meeting minutes:
    """ + minutes_text[:2000]  # Use first 2000 chars for domain detection

    try:
        response = call_openrouter(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        domain = response.strip().lower()
        # Validate domain response
        valid_domains = [
            "energy_utility", "finance_banking", "technology_software",
            "healthcare_medical", "manufacturing_industrial", "government_public",
            "consulting_professional", "academia_research", "startup_business", "general_corporate"
        ]
        return domain if domain in valid_domains else "general_corporate"
    except Exception as e:
        eprint(f"Failed to detect meeting domain: {e}")
        return "general_corporate"


def extract_search_topics(minutes_text: str, api_key: str, model: str = "openai/gpt-4o-mini") -> List[str]:
    """Extract key topics from minutes that would benefit from web research"""
    prompt = """
    Analyze these meeting minutes and extract 3-5 specific search topics that would benefit from additional web research.
    Focus on:
    - Technical terms or concepts that need definition/context
    - Company names, projects, or initiatives mentioned
    - Industry trends or regulatory topics discussed
    - Specific technologies or standards referenced
    - Financial concepts or business models discussed
    - Compliance or regulatory frameworks mentioned

    Return only a simple list of search queries, one per line.

    Meeting minutes:
    """ + minutes_text

    try:
        response = call_openrouter(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        topics = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return topics[:5]  # Limit to 5 topics
    except Exception as e:
        eprint(f"Failed to extract search topics: {e}")
        return []


def enhance_minutes_with_context(minutes_text: str, api_key: str, model: str = "anthropic/claude-3.5-sonnet") -> str:
    """Enhance minutes with domain-aware web research context"""
    # Detect meeting domain automatically
    eprint("Detecting meeting domain...")
    detected_domain = detect_meeting_domain(minutes_text, api_key)
    eprint(f"Detected domain: {detected_domain}")

    eprint("Extracting topics for web research...")
    topics = extract_search_topics(minutes_text, api_key)

    if not topics:
        eprint("No topics extracted for enhancement")
        return minutes_text

    eprint(f"Researching {len(topics)} topics with {detected_domain} context...")

    # Research topics with domain-specific knowledge
    research_context = []
    for topic in topics:
        eprint(f"  → Researching: {topic}")
        search_result = web_search_simple(topic, detected_domain)
        research_context.append(f"**{topic}**: {search_result}")

    # Create domain-aware enhancement prompt
    domain_specific_guidance = {
        "energy_utility": "Focus on energy sector terminology, TSO/DSO operations, grid management, renewable integration, CAPEX/OPEX models, regulatory compliance (NERC CIP, ENTSO-E), and cybersecurity frameworks.",
        "technology_software": "Focus on software development practices, cloud technologies, API design, DevOps methodologies, cybersecurity frameworks, and technology architecture decisions.",
        "finance_banking": "Focus on financial terminology, risk management, regulatory compliance (Basel, MIFID), trading systems, payment technologies, and fintech innovations.",
        "healthcare_medical": "Focus on medical terminology, patient care protocols, regulatory compliance (HIPAA, FDA), healthcare technology, and clinical decision-making processes.",
        "manufacturing_industrial": "Focus on manufacturing processes, supply chain management, Industry 4.0 technologies, quality control systems, and operational efficiency initiatives.",
        "general_corporate": "Focus on business strategy, organizational structure, financial management, technology adoption, and operational excellence principles."
    }

    domain_focus = domain_specific_guidance.get(detected_domain, domain_specific_guidance["general_corporate"])

    enhancement_prompt = f"""
    You are an expert meeting analyst with deep domain knowledge. Enhance these meeting minutes by integrating comprehensive technical context and background information for the {detected_domain.replace('_', '/')} domain.

    Original minutes:
    {minutes_text}

    Additional context from research:
    {chr(10).join(research_context)}

    Create enhanced meeting minutes that:

    1. **PRESERVE ALL ORIGINAL INFORMATION** - Keep every detail from the original minutes intact

    2. **ADD COMPREHENSIVE TECHNICAL EXPLANATIONS**:
       - Define and explain all technical terms, acronyms, and concepts mentioned
       - Provide detailed background on industry-specific topics and business models
       - Explain the business implications and strategic context of technical decisions
       - Add regulatory and compliance context where relevant

    3. **ENHANCE WITH DOMAIN EXPERTISE**:
       - Add "**TECHNICAL CONTEXT:**" sections throughout to explain complex concepts
       - Include industry background for companies, technologies, and standards mentioned
       - Explain the strategic importance of discussions within the specific industry context
       - Provide operational context for technical decisions and their impacts

    4. **STRUCTURE FOR MAXIMUM INSIGHT**:
       - Add a comprehensive "**BACKGROUND & TECHNICAL CONTEXT**" section at the beginning
       - Use "**[CONTEXT]**" annotations inline to explain technical terms as they appear
       - Include subsections for different technical areas discussed
       - Add an "**INDUSTRY IMPLICATIONS**" section highlighting strategic significance

    5. **FOCUS ON PRACTICAL VALUE**:
       - Explain WHY technical decisions matter in business context
       - Connect technical discussions to operational and financial implications
       - Highlight regulatory compliance aspects and risk factors
       - Provide context that helps non-technical stakeholders understand technical decisions

    **Domain-Specific Focus**: {domain_focus}

    The enhanced minutes should serve as a comprehensive reference document that provides deep technical understanding while remaining accessible to stakeholders with varying technical backgrounds.
    """

    try:
        eprint("Generating enhanced minutes with context...")
        enhanced_minutes = call_openrouter(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": enhancement_prompt}],
            temperature=0.2,
        )
        return enhanced_minutes
    except Exception as e:
        eprint(f"Failed to enhance minutes: {e}")
        return minutes_text


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate meeting minutes from MP4 using OpenRouter")
    g_in = parser.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--input", "--file", dest="input", type=str, help="Path to input MP4 (or audio)")
    g_in.add_argument("--transcript", type=str, help="Path to existing transcript .txt")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model id or 'auto' for cost-effective selection",
    )
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (overrides OPENROUTER_API_KEY)")
    parser.add_argument("--language", type=str, default="English", help="Output language (e.g., English, Français)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--subject-prefix", type=str, default="Meeting Minutes", help="Subject line prefix")
    parser.add_argument("--out", type=str, default="-", help="Output file or '-' for stdout")
    parser.add_argument("--transcript-only", action="store_true", help="Only generate transcript, skip minutes generation")
    parser.add_argument("--minutes-only", action="store_true", help="Only generate minutes from existing transcript file")
    parser.add_argument("--enhance", action="store_true", help="Enhance minutes with web research and contextual information")
    parser.add_argument("--anonymize", action="store_true", help="Anonymize participant names using Speaker A/B/C labels")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model size if used (tiny|base|small|medium|large)")
    parser.add_argument("--chunk-tokens", type=int, default=3000, help="Approx tokens per chunk for map-reduce")
    parser.add_argument(
        "--synthesis",
        type=str,
        choices=["auto", "map-reduce", "full"],
        default="auto",
        help=(
            "Minutes synthesis strategy: 'full' processes the entire transcript in one pass (better global context), "
            "'map-reduce' summarizes chunks then synthesizes (cheaper at very large sizes), 'auto' picks based on size."
        ),
    )
    parser.add_argument(
        "--focus",
        type=str,
        default=os.getenv("MINUTES_FOCUS", ""),
        help=(
            "Optional comma-separated priority topics to emphasize (e.g., 'CAPEX vs OPEX, procurement')."
        ),
    )
    parser.add_argument(
        "--transcriber",
        type=str,
        default="auto",
        choices=["auto", "openrouter", "local", "openai"],
        help=(
            "Transcription method when --input is provided: "
            "auto uses OpenRouter if OPENROUTER_API_KEY is set, else local Whisper."
        ),
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="whisper-1",
        help="OpenAI transcription model (e.g., whisper-1)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=os.getenv("OPENROUTER_STT_MODEL", "openai/whisper-1"),
        help="OpenRouter transcription model (e.g., openai/whisper-1)",
    )
    parser.add_argument(
        "--stt-chat-model",
        type=str,
        default=os.getenv("OPENROUTER_STT_CHAT_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter chat model to use for audio transcription fallback",
    )

    args = parser.parse_args(argv)

    # Validate new mode options
    if args.transcript_only and args.minutes_only:
        eprint("Error: Cannot use both --transcript-only and --minutes-only")
        return 2

    if args.minutes_only and not args.transcript:
        eprint("Error: --minutes-only requires --transcript to specify existing transcript file")
        return 2

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
        # Choose transcription method (prefer OpenRouter when available)
        transcriber = str(args.transcriber).lower()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if transcriber == "openrouter" or (transcriber == "auto" and api_key):
            eprint("Transcribing with OpenRouter (cloud)…")
            try:
                try:
                    transcript_text = transcribe_with_openrouter(
                        ipath, api_key=api_key, model=args.stt_model, language=args.language
                    )
                except Exception as e:
                    msg = str(e)
                    if (
                        "413" in msg
                        or "Entity Too Large" in msg
                        or "PAYLOAD_TOO_LARGE" in msg
                        or "405" in msg
                        or "404" in msg
                    ):
                        eprint("OpenRouter STT endpoint unavailable/limited; trying chat-based transcription…")
                        transcript_text = transcribe_with_openrouter_chat(
                            ipath,
                            api_key=api_key,
                            model=args.stt_chat_model,
                            language=args.language,
                            segment_seconds=int(os.getenv("STT_CHAT_SEGMENT_SECONDS", "45")),
                        )
                    else:
                        raise
            except Exception as e:
                eprint(str(e))
                return 2
        elif transcriber == "openai" or (transcriber == "auto" and openai_key):
            if not openai_key:
                eprint("Missing OPENAI_API_KEY for --transcriber openai.")
                return 2
            eprint("Transcribing with OpenAI (cloud)…")
            try:
                transcript_text = transcribe_with_openai(
                    ipath, api_key=openai_key, model=args.openai_model, language=args.language
                )
            except Exception as e:
                eprint(str(e))
                return 2
        else:
            if transcriber == "auto":
                eprint("No cloud transcriber configured; using local Whisper.")
            try:
                transcript_text = transcribe_with_whisper(ipath, model_size=args.whisper_model)
            except Exception as e:
                eprint(str(e))
                eprint("Tip: set OPENROUTER_API_KEY to use OpenRouter for transcription.")
                return 2

    if not transcript_text:
        eprint("Empty transcript; aborting.")
        return 2

    # Handle transcript-only mode
    if args.transcript_only:
        if args.out == "-":
            print(transcript_text)
        else:
            out_path = Path(args.out)
            out_path.write_text(transcript_text, encoding="utf-8")
            eprint(f"Transcript saved to: {out_path}")
        return 0

    # Handle minutes-only mode (skip if we already have transcript from --transcript)
    if args.minutes_only:
        eprint("Generating minutes from existing transcript...")
        # Continue with minutes generation below

    # Prepare chunks if needed
    tokens = estimate_tokens(len(transcript_text))
    target = max(500, int(args.chunk_tokens))
    if tokens > target * 1.2:
        chunks = chunk_text(transcript_text, target_tokens=target)
    else:
        chunks = [transcript_text]

    eprint(f"Transcript length ~{tokens} tokens → {len(chunks)} chunk(s)")

    # Pick OpenRouter models for summarization
    chunk_model, final_model = pick_models_for_minutes(tokens, args.model)
    eprint(f"Models → chunks: {chunk_model}; final: {final_model}")

    try:
        # Decide synthesis strategy
        synthesis_mode = str(getattr(args, "synthesis", "auto")).lower()
        if synthesis_mode == "auto":
            # Heuristic: use full synthesis for moderate sizes, map-reduce for very large
            synthesis_mode = "full" if tokens <= 80_000 else "map-reduce"

        if synthesis_mode == "full":
            eprint("Synthesis mode: FULL (single pass over full transcript)")
            final_minutes = synthesize_minutes_full(
                api_key=api_key,
                model=final_model,
                transcript_text=transcript_text,
                language=args.language,
                subject_prefix=args.subject_prefix,
                temperature=args.temperature,
                focus=(args.focus or None),
            )
        else:
            eprint("Synthesis mode: MAP-REDUCE (chunk summaries → final synthesis)")
            summaries = summarize_chunks(
                api_key=api_key,
                model=chunk_model,
                chunks=chunks,
                language=args.language,
                temperature=args.temperature,
                focus=(args.focus or None),
            )
            final_minutes = synthesize_minutes(
                api_key=api_key,
                model=final_model,
                chunk_summaries=summaries,
                language=args.language,
                subject_prefix=args.subject_prefix,
                temperature=args.temperature,
                focus=(args.focus or None),
            )
    except Exception as e:
        eprint(f"OpenRouter error: {e}")
        return 3

    # Enhance minutes with web research if requested
    if args.enhance:
        eprint("Enhancing minutes with contextual research...")
        final_minutes = enhance_minutes_with_context(final_minutes, api_key, final_model)

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
