"""KittenTTS Studio — Flask API Server"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import threading
import uuid
from datetime import datetime
from queue import Queue

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
TRASH_DIR = os.path.join(AUDIO_DIR, "TRASH")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)

MODELS = {
    "mini": {
        "name": "Kitten TTS Mini",
        "params": "80M",
        "size": "80MB",
        "repo": "KittenML/kitten-tts-mini-0.8",
    },
    "micro": {
        "name": "Kitten TTS Micro",
        "params": "40M",
        "size": "41MB",
        "repo": "KittenML/kitten-tts-micro-0.8",
    },
    "nano": {
        "name": "Kitten TTS Nano",
        "params": "15M",
        "size": "56MB",
        "repo": "KittenML/kitten-tts-nano-0.8",
    },
    "nano-int8": {
        "name": "Kitten TTS Nano INT8",
        "params": "15M",
        "size": "19MB",
        "repo": "KittenML/kitten-tts-nano-0.8-int8",
    },
    "nano-fp32": {
        "name": "Kitten TTS Nano FP32",
        "params": "15M",
        "size": "~56MB",
        "repo": "KittenML/kitten-tts-nano-0.8-fp32",
    },
}

VOICES = ["Rosie", "Hugo", "Bella", "Bruno", "Jasper", "Luna", "Kiki", "Leo"]

# Cache of loaded KittenTTS model instances: {model_id: KittenTTS}
loaded_models = {}
model_lock = threading.Lock()

# Alignment model (stable-ts / Whisper) — optional feature
alignment_model = None
alignment_lock = threading.Lock()
alignment_available = None  # None = not checked yet, True/False after first check
alignment_tasks = {}        # {basename: threading.Thread}
alignment_tasks_lock = threading.Lock()

# Enhancement model (LavaSR) — optional feature
enhance_model = None
enhance_lock = threading.Lock()
enhance_available = None  # None = not checked yet, True/False after first check
enhance_tasks = {}        # {basename: threading.Thread}
enhance_tasks_lock = threading.Lock()

# Silence removal model (Silero VAD) — optional feature
vad_model = None
vad_utils = None
vad_lock = threading.Lock()
vad_available = None  # None = not checked yet, True/False after first check
vad_tasks = {}        # {basename: threading.Thread}
vad_tasks_lock = threading.Lock()

# Chunked generation jobs: {job_id: {"queue": Queue, "status": str, "metadata": dict, "created": float}}
generation_jobs = {}
generation_jobs_lock = threading.Lock()
generation_inference_lock = threading.Lock()  # Serialize ONNX inference (not thread-safe)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_filename(prompt: str) -> str:
    excerpt = re.sub(r"[^a-zA-Z0-9]+", "-", prompt[:30].lower()).strip("-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{excerpt}_{timestamp}"


def clean_for_tts(text: str) -> str:
    """Strip markdown formatting, URLs, and excess whitespace before TTS."""
    text = re.sub(r"[*_#`~]", "", text)       # markdown chars
    text = re.sub(r"https?://\S+", "link", text)  # URLs
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# TTS Text Normalization
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
    "I'd": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "it'd": "it would", "it'll": "it will", "it's": "it is",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
    "that's": "that is", "that'd": "that would", "that'll": "that will",
    "who's": "who is", "who'd": "who would", "who'll": "who will",
    "what's": "what is", "what'd": "what did", "what'll": "what will",
    "where's": "where is", "when's": "when is", "why's": "why is", "how's": "how is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
    "let's": "let us", "there's": "there is", "here's": "here is", "o'clock": "of the clock",
}

_ORDINALS = {
    r'\b1st\b': 'first', r'\b2nd\b': 'second', r'\b3rd\b': 'third',
    r'\b4th\b': 'fourth', r'\b5th\b': 'fifth', r'\b6th\b': 'sixth',
    r'\b7th\b': 'seventh', r'\b8th\b': 'eighth', r'\b9th\b': 'ninth',
    r'\b10th\b': 'tenth', r'\b11th\b': 'eleventh', r'\b12th\b': 'twelfth',
    r'\b13th\b': 'thirteenth', r'\b14th\b': 'fourteenth', r'\b15th\b': 'fifteenth',
    r'\b20th\b': 'twentieth', r'\b21st\b': 'twenty first', r'\b22nd\b': 'twenty second',
    r'\b23rd\b': 'twenty third', r'\b30th\b': 'thirtieth', r'\b31st\b': 'thirty first',
}

_ABBREVIATIONS = {
    r'\bDr\.\b': 'Doctor', r'\bMr\.\b': 'Mister', r'\bMrs\.\b': 'Missus', r'\bMs\.\b': 'Miss',
    r'\bProf\.\b': 'Professor', r'\bSt\.\b': 'Saint', r'\bAve\.\b': 'Avenue',
    r'\bBlvd\.\b': 'Boulevard', r'\bDept\.\b': 'Department', r'\bEst\.\b': 'Estimated',
    r'\betc\.\b': 'et cetera', r'\be\.g\.\b': 'for example', r'\bi\.e\.\b': 'that is',
    r'\bvs\.\b': 'versus', r'\bapprox\.\b': 'approximately',
    r'\bmin\.\b': 'minutes', r'\bmax\.\b': 'maximum', r'\bno\.\b': 'number',
    r'\bAPI\b': 'A P I', r'\bURL\b': 'U R L', r'\bHTTP\b': 'H T T P',
    r'\bHTML\b': 'H T M L', r'\bCSS\b': 'C S S', r'\bSQL\b': 'S Q L',
    r'\bRBQ\b': 'R B Q', r'\bID\b': 'I D', r'\bPIN\b': 'pin',
    r'\bOTP\b': 'O T P', r'\bSMS\b': 'S M S', r'\bPDF\b': 'P D F',
}

_UNITS = {
    r'(\d+)\s?km\b': r'\1 kilometers', r'(\d+)\s?m\b': r'\1 meters',
    r'(\d+)\s?cm\b': r'\1 centimeters', r'(\d+)\s?mm\b': r'\1 millimeters',
    r'(\d+)\s?kg\b': r'\1 kilograms', r'(\d+)\s?g\b': r'\1 grams',
    r'(\d+)\s?mg\b': r'\1 milligrams', r'(\d+)\s?lb\b': r'\1 pounds',
    r'(\d+)\s?oz\b': r'\1 ounces', r'(\d+)\s?mph\b': r'\1 miles per hour',
    r'(\d+)\s?kph\b': r'\1 kilometers per hour',
    r'(\d+)\s?°C\b': r'\1 degrees Celsius', r'(\d+)\s?°F\b': r'\1 degrees Fahrenheit',
    r'(\d+)\s?%': r'\1 percent',
    r'(\d+)\s?MB\b': r'\1 megabytes', r'(\d+)\s?GB\b': r'\1 gigabytes',
    r'(\d+)\s?TB\b': r'\1 terabytes', r'(\d+)\s?ms\b': r'\1 milliseconds',
    r'(\d+)\s?fps\b': r'\1 frames per second',
}

_SYMBOLS = {
    '&': 'and', '@': 'at', '#': 'number', '+': 'plus', '=': 'equals',
    '>': 'greater than', '<': 'less than', '~': 'approximately',
    '|': '', '\\': '', '/': ' or ',
    '\u2013': '-', '\u2014': ',', '\u2026': '...',
    '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
}

_DATE_MONTHS = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December',
}


def _expand_symbols(text):
    for sym, rep in _SYMBOLS.items():
        text = text.replace(sym, rep)
    return text

def _expand_contractions(text):
    for c, e in _CONTRACTIONS.items():
        text = re.sub(re.escape(c), e, text, flags=re.IGNORECASE)
    return text

def _expand_abbreviations(text):
    for p, r in _ABBREVIATIONS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_currency(text):
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)
    text = re.sub(r'€(\d+)', r'\1 euros', text)
    text = re.sub(r'£(\d+)', r'\1 pounds', text)
    text = re.sub(r'¥(\d+)', r'\1 yen', text)
    text = re.sub(r'HTG\s?(\d+)', r'\1 Haitian gourdes', text)
    return text

def _expand_units(text):
    for p, r in _UNITS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_dates(text):
    def _replace(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{_DATE_MONTHS.get(mo, mo)} {int(d)}, {y}"
    return re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', _replace, text)

def _expand_time(text):
    text = re.sub(
        r'\b(\d{1,2}):(\d{2})\s?(am|pm)\b',
        lambda m: f"{m.group(1)} {m.group(2)} {m.group(3).replace('am','a m').replace('pm','p m')}",
        text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(\d{1,2})\s?(am|pm)\b',
        lambda m: f"{m.group(1)} {m.group(2).replace('am','a m').replace('pm','p m')}",
        text, flags=re.IGNORECASE)
    return text

def _expand_ordinals(text):
    for p, r in _ORDINALS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_numbers(text):
    try:
        from num2words import num2words
        # Floats first (3.14 -> three point one four)
        def _float_repl(m):
            whole = num2words(int(m.group(1)))
            decimals = " ".join(num2words(int(d)) for d in m.group(2))
            return f"{whole} point {decimals}"
        text = re.sub(r'\b(\d+)\.(\d+)\b', _float_repl, text)
        # Integers (42 -> forty-two)
        text = re.sub(r'\b(\d+)\b', lambda m: num2words(int(m.group(1))), text)
    except ImportError:
        pass  # num2words not installed — skip number expansion
    return text


def normalize_for_tts(text: str) -> str:
    """Full TTS normalization pipeline. Order matters."""
    text = _expand_symbols(text)
    text = _expand_contractions(text)
    text = _expand_abbreviations(text)
    text = _expand_currency(text)
    text = _expand_units(text)
    text = _expand_dates(text)
    text = _expand_time(text)
    text = _expand_ordinals(text)
    text = _expand_numbers(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting & audio concatenation for chunked generation
# ---------------------------------------------------------------------------

_SENTENCE_ABBREVS = {
    'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'St', 'Ave', 'Blvd', 'Dept',
    'Est', 'etc', 'vs', 'approx', 'Jr', 'Sr', 'Gen', 'Gov', 'Sgt',
    'Cpl', 'Pvt', 'Lt', 'Capt', 'Maj', 'Col', 'Inc', 'Corp', 'Ltd',
    'Co', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep',
    'Oct', 'Nov', 'Dec', 'Fig', 'Vol', 'Ref', 'no', 'No',
}
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _SENTENCE_ABBREVS) + r')\.',
    re.IGNORECASE,
)

MAX_SENTENCE_LEN = 150


def split_into_sentences(text: str, max_len: int = MAX_SENTENCE_LEN) -> list[str]:
    """Split text into sentence-length chunks suitable for TTS.

    1. Protect abbreviations, ellipses, and decimal numbers from splitting.
    2. Split on sentence-ending punctuation (.!?) followed by whitespace.
    3. Sub-split any chunk exceeding *max_len* on ; then , then word boundaries.
    4. Ensure every chunk ends with punctuation.
    """
    if not text or not text.strip():
        return []

    # --- Protect patterns that should NOT be treated as sentence boundaries ---
    ABBREV_PLACEHOLDER = '\x00'
    ELLIPSIS_PLACEHOLDER = '\x01'
    DECIMAL_PLACEHOLDER = '\x02'

    # Protect abbreviations: "Dr." → "Dr\x00"
    protected = _ABBREV_PATTERN.sub(lambda m: m.group(1) + ABBREV_PLACEHOLDER, text)
    # Protect ellipsis: "..." → "\x01"
    protected = protected.replace('...', ELLIPSIS_PLACEHOLDER)
    # Protect decimal numbers: "3.14" → "3\x0214"
    protected = re.sub(r'(\d)\.(\d)', r'\1' + DECIMAL_PLACEHOLDER + r'\2', protected)

    # --- Split on sentence-ending punctuation followed by space or end-of-string ---
    raw_parts = re.split(r'(?<=[.!?])\s+', protected)

    # --- Restore placeholders ---
    def restore(s: str) -> str:
        s = s.replace(ABBREV_PLACEHOLDER, '.')
        s = s.replace(ELLIPSIS_PLACEHOLDER, '...')
        s = s.replace(DECIMAL_PLACEHOLDER, '.')
        return s.strip()

    sentences = [restore(p) for p in raw_parts if p.strip()]

    # --- Sub-split long sentences ---
    def _subsplit(s: str, limit: int) -> list[str]:
        if len(s) <= limit:
            return [s]
        # Try splitting on semicolons
        parts = [p.strip() for p in s.split(';') if p.strip()]
        if len(parts) > 1 and all(len(p) <= limit for p in parts):
            return parts
        # Try splitting on commas
        result = []
        for part in parts:
            if len(part) <= limit:
                result.append(part)
            else:
                sub = [p.strip() for p in part.split(',') if p.strip()]
                if len(sub) > 1:
                    result.extend(sub)
                else:
                    result.append(part)
        # Final pass: word-boundary split for anything still too long
        final = []
        for part in result:
            if len(part) <= limit:
                final.append(part)
            else:
                words = part.split()
                buf = ''
                for w in words:
                    if buf and len(buf) + 1 + len(w) > limit:
                        final.append(buf)
                        buf = w
                    else:
                        buf = (buf + ' ' + w) if buf else w
                if buf:
                    final.append(buf)
        return final

    chunks = []
    for s in sentences:
        chunks.extend(_subsplit(s, max_len))

    # --- Ensure punctuation on every chunk ---
    def _ensure_punct(s: str) -> str:
        s = s.strip()
        if s and s[-1] not in '.!?,;:':
            s += ','
        return s

    return [_ensure_punct(c) for c in chunks if c.strip()]


def concatenate_chunks(chunks: list, sample_rate: int = 24000,
                       gap_ms: int = 80, crossfade_ms: int = 20) -> np.ndarray:
    """Concatenate audio chunks with silence gaps and crossfade.

    Args:
        chunks: List of 1-D or 2-D numpy audio arrays.
        sample_rate: Audio sample rate (default 24 kHz).
        gap_ms: Silence between sentences in milliseconds.
        crossfade_ms: Linear crossfade at boundaries in milliseconds.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    # Flatten any 2-D arrays (some ONNX models return [1, N])
    flat = [c.squeeze() for c in chunks]
    if len(flat) == 1:
        return flat[0]

    gap_samples = int(sample_rate * gap_ms / 1000)
    xfade_samples = int(sample_rate * crossfade_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    parts = []
    for i, chunk in enumerate(flat):
        if i == 0:
            parts.append(chunk)
            continue
        prev = parts[-1]
        # Apply crossfade if both chunks are long enough
        if xfade_samples > 0 and len(prev) >= xfade_samples and len(chunk) >= xfade_samples:
            fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)
            tail = prev[-xfade_samples:] * fade_out
            head = chunk[:xfade_samples] * fade_in
            # Replace tail of previous part with crossfaded overlap
            parts[-1] = prev[:-xfade_samples]
            parts.append(tail + head)
            parts.append(silence)
            parts.append(chunk[xfade_samples:])
        else:
            parts.append(silence)
            parts.append(chunk)

    return np.concatenate(parts)


def find_available_port(start: int = 5000) -> int:
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start


def is_model_cached(repo_id: str) -> bool:
    """Check if config.json is already in the HF cache (quick proxy check)."""
    result = try_to_load_from_cache(repo_id, "config.json")
    return result is not None and not isinstance(result, str) is False


def get_model_files(repo_id: str) -> list[str]:
    """Download config.json (usually cached) and return [config, model_file, voices]."""
    config_path = try_to_load_from_cache(repo_id, "config.json")
    if config_path and isinstance(config_path, str):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return ["config.json", cfg["model_file"], cfg["voices"]]
    return ["config.json"]


def load_model(model_id: str):
    """Load (or return cached) KittenTTS model instance."""
    if model_id in loaded_models:
        return loaded_models[model_id]

    repo = MODELS[model_id]["repo"]
    from kittentts import KittenTTS

    with model_lock:
        if model_id not in loaded_models:
            loaded_models[model_id] = KittenTTS(repo)
    return loaded_models[model_id]


# ---------------------------------------------------------------------------
# SSE Progress tqdm
# ---------------------------------------------------------------------------


class SSEProgressCapture(tqdm):
    """Custom tqdm that pushes progress events to a Queue for SSE streaming."""

    progress_queue: Queue | None = None

    def __init__(self, *args, **kwargs):
        self._sse_queue = kwargs.pop("sse_queue", None) or getattr(
            SSEProgressCapture, "progress_queue", None
        )
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self._sse_queue is None:
            return
        total = self.total or 0
        downloaded = self.n or 0
        progress = int((downloaded / total) * 100) if total else 0
        speed = self.format_dict.get("rate", 0) or 0

        if speed >= 1_000_000:
            speed_str = f"{speed / 1_000_000:.1f}MB/s"
        elif speed >= 1_000:
            speed_str = f"{speed / 1_000:.1f}KB/s"
        else:
            speed_str = f"{speed:.0f}B/s"

        if total >= 1_000_000:
            size_str = f"{total / 1_000_000:.2f}MB"
        elif total >= 1_000:
            size_str = f"{total / 1_000:.1f}KB"
        else:
            size_str = f"{total}B"

        event = {
            "phase": "downloading",
            "file": self.desc or "unknown",
            "progress": progress,
            "downloaded_mb": round(downloaded / 1_000_000, 2),
            "total_mb": round(total / 1_000_000, 2),
            "size": size_str,
            "speed": speed_str,
        }
        self._sse_queue.put(event)


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


# --- Serve frontend ---
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# --- Health ---
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "port": request.host.split(":")[-1], "ffmpeg": _find_ffmpeg() is not None, "alignment": _check_alignment_available(), "enhance": _check_enhance_available(), "vad": _check_vad_available()})


# --- Normalize text ---
@app.route("/api/normalize", methods=["POST"])
def normalize_text():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    normalized = normalize_for_tts(text)
    return jsonify({"original": text, "normalized": normalized})


# --- Models ---
@app.route("/api/models")
def models():
    out = []
    for mid, m in MODELS.items():
        out.append({"id": mid, **m})
    return jsonify(out)


# --- Voices ---
@app.route("/api/voices")
def voices():
    return jsonify(VOICES)


# --- Model status ---
@app.route("/api/model-status/<model_id>")
def model_status(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    repo = MODELS[model_id]["repo"]
    cached = try_to_load_from_cache(repo, "config.json")
    is_cached = cached is not None and isinstance(cached, str)

    files = []
    if is_cached:
        files = get_model_files(repo)
    return jsonify({"model_id": model_id, "cached": is_cached, "files": files})


# --- Download model with SSE progress ---
@app.route("/api/download-model/<model_id>")
def download_model(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404

    repo = MODELS[model_id]["repo"]

    def _download_file(repo, filename, result):
        """Run hf_hub_download in a thread, storing result or exception."""
        try:
            path = hf_hub_download(
                repo_id=repo, filename=filename, tqdm_class=SSEProgressCapture
            )
            result["path"] = path
        except Exception as e:
            result["error"] = e

    def _stream_download(repo, filename, q):
        """Start download in thread, yield SSE events as they arrive."""
        result = {}
        t = threading.Thread(target=_download_file, args=(repo, filename, result))
        t.start()
        while t.is_alive():
            t.join(timeout=0.15)
            while not q.empty():
                yield f"data: {json.dumps(q.get())}\n\n"
        # Drain remaining events
        while not q.empty():
            yield f"data: {json.dumps(q.get())}\n\n"
        if "error" in result:
            raise result["error"]

    def stream():
        q = Queue()
        yield f"data: {json.dumps({'phase': 'checking', 'model': model_id})}\n\n"

        try:
            SSEProgressCapture.progress_queue = q

            # Step 1: download config.json
            for event in _stream_download(repo, "config.json", q):
                yield event

            config_path = try_to_load_from_cache(repo, "config.json")
            if not isinstance(config_path, str):
                raise RuntimeError("Failed to download config.json")

            # Step 2: read config to get model_file and voices filenames
            with open(config_path, "r") as f:
                cfg = json.load(f)
            model_file = cfg["model_file"]
            voices_file = cfg["voices"]

            # Step 3: download model ONNX
            for event in _stream_download(repo, model_file, q):
                yield event

            # Step 4: download voices
            for event in _stream_download(repo, voices_file, q):
                yield event

            SSEProgressCapture.progress_queue = None

            # Step 5: load model into memory
            yield f"data: {json.dumps({'phase': 'loading', 'message': 'Loading model...'})}\n\n"
            load_model(model_id)

            yield f"data: {json.dumps({'phase': 'ready', 'message': 'Model ready'})}\n\n"

        except Exception as e:
            SSEProgressCapture.progress_queue = None
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Chunked generation background worker ---

def _background_chunked_generate(job_id, model_id, voice, sentences, speed,
                                  max_silence_ms, prompt, basename):
    """Generate audio for each sentence chunk, concatenate, loudnorm, and save."""
    q = generation_jobs[job_id]["queue"]
    try:
        m = load_model(model_id)
        inner = m.model  # KittenTTS_1_Onnx instance

        audio_chunks = []
        total = len(sentences)
        total_inference = 0.0

        for i, sentence in enumerate(sentences):
            q.put({"phase": "generating", "chunk": i + 1, "total": total,
                    "sentence": sentence})

            # Preprocess the same way KittenTTS.generate() does internally
            processed = inner.preprocessor(sentence)
            # Ensure trailing punctuation
            processed = processed.strip()
            if processed and processed[-1] not in '.!?,;:':
                processed += ','

            start = time.perf_counter()
            with generation_inference_lock:
                chunk_audio = inner.generate_single_chunk(processed, voice=voice, speed=speed)
            elapsed = time.perf_counter() - start
            total_inference += elapsed
            audio_chunks.append(chunk_audio)

        # Concatenate with crossfade and silence gaps
        q.put({"phase": "concatenating"})
        audio = concatenate_chunks(audio_chunks, sample_rate=24000, gap_ms=80, crossfade_ms=20)

        wav_path = os.path.join(AUDIO_DIR, basename + ".wav")
        sf.write(wav_path, audio, 24000)

        # Apply loudnorm
        q.put({"phase": "normalizing"})
        _run_loudnorm(wav_path)

        # Re-read to get accurate duration after loudnorm
        info = sf.info(wav_path)
        duration_generated = info.duration
        rtf = total_inference / duration_generated if duration_generated > 0 else 0

        words = len(prompt.split())
        approx_tokens = int(words * 1.3)

        metadata = {
            "filename": basename + ".wav",
            "prompt": prompt.strip(),
            "model": MODELS[model_id]["repo"],
            "model_id": model_id,
            "voice": voice,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inference_time": round(total_inference, 3),
            "rtf": round(rtf, 4),
            "duration_seconds": round(duration_generated, 2),
            "sample_rate": 24000,
            "speed": speed,
            "max_silence_ms": max_silence_ms,
            "words": words,
            "approx_tokens": approx_tokens,
            "chunked": True,
            "num_chunks": total,
        }
        json_path = os.path.join(AUDIO_DIR, basename + ".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Set post-processing statuses and write to disk BEFORE starting threads
        metadata["alignment_status"] = "pending" if _check_alignment_available() else "unavailable"
        metadata["enhance_status"] = "pending" if _check_enhance_available() else "unavailable"
        metadata["vad_status"] = "pending" if _check_vad_available() else "unavailable"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Now kick off background post-processing
        _start_alignment(basename)
        _start_enhancement(basename)
        if not _check_enhance_available():
            _start_vad(basename, max_silence_ms)

        q.put({"phase": "done", "metadata": metadata})
        with generation_jobs_lock:
            generation_jobs[job_id]["status"] = "done"
            generation_jobs[job_id]["metadata"] = metadata

    except Exception as e:
        print(f"[chunked-generate] Error: {e}")
        q.put({"phase": "error", "message": str(e)})
        with generation_jobs_lock:
            generation_jobs[job_id]["status"] = "error"


def _cleanup_old_jobs(max_age_s=300):
    """Remove generation jobs older than max_age_s seconds."""
    now = time.time()
    with generation_jobs_lock:
        expired = [jid for jid, job in generation_jobs.items()
                   if now - job.get("created", 0) > max_age_s]
        for jid in expired:
            del generation_jobs[jid]


# --- Generate audio ---
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model_id = data.get("model", "mini")
    voice = data.get("voice", "Jasper")
    prompt = data.get("prompt", "")
    speed = float(data.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))  # clamp to 0.5–2.0
    max_silence_ms = int(data.get("max_silence_ms", 500))
    max_silence_ms = max(200, min(1000, max_silence_ms))  # clamp to 200–1000

    if not prompt.strip():
        return jsonify({"error": "Prompt is required"}), 400
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice. Choose from: {VOICES}"}), 400

    m = load_model(model_id)

    tts_prompt = clean_for_tts(prompt)
    sentences = split_into_sentences(tts_prompt)

    # --- Multi-sentence: chunked background generation with SSE progress ---
    if len(sentences) > 1:
        _cleanup_old_jobs()
        job_id = uuid.uuid4().hex[:12]
        basename = generate_filename(prompt)
        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "queue": Queue(),
                "status": "running",
                "metadata": None,
                "created": time.time(),
            }
        t = threading.Thread(
            target=_background_chunked_generate,
            args=(job_id, model_id, voice, sentences, speed,
                  max_silence_ms, prompt, basename),
            daemon=True,
        )
        t.start()
        return jsonify({
            "job_id": job_id,
            "status": "chunking",
            "total_chunks": len(sentences),
            "sentences": sentences,
        }), 202

    # --- Single sentence: synchronous fast path (unchanged) ---
    words = len(tts_prompt.split())
    approx_tokens = int(words * 1.3)

    start = time.perf_counter()
    try:
        audio = m.generate(tts_prompt, voice=voice, speed=speed)
    except Exception as e:
        print(f"[generate] TTS inference failed: {e}")
        return jsonify({"error": f"Generation failed: {e}"}), 500
    end = time.perf_counter()

    duration_generated = len(audio) / 24000
    inference_time = end - start
    rtf = inference_time / duration_generated

    basename = generate_filename(prompt)
    wav_name = f"{basename}.wav"
    json_name = f"{basename}.json"

    sf.write(os.path.join(AUDIO_DIR, wav_name), audio, 24000)

    metadata = {
        "filename": wav_name,
        "prompt": prompt.strip(),
        "model": MODELS[model_id]["repo"],
        "model_id": model_id,
        "voice": voice,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inference_time": round(inference_time, 3),
        "rtf": round(rtf, 4),
        "duration_seconds": round(duration_generated, 2),
        "sample_rate": 24000,
        "speed": speed,
        "max_silence_ms": max_silence_ms,
        "words": words,
        "approx_tokens": approx_tokens,
    }
    with open(os.path.join(AUDIO_DIR, json_name), "w") as f:
        json.dump(metadata, f, indent=2)

    # Kick off background alignment and enhancement
    _start_alignment(basename)
    metadata["alignment_status"] = "pending" if _check_alignment_available() else "unavailable"
    _start_enhancement(basename)
    metadata["enhance_status"] = "pending" if _check_enhance_available() else "unavailable"
    # VAD is chained from enhancement; if enhance unavailable, start VAD directly
    if not _check_enhance_available():
        _start_vad(basename, max_silence_ms)
    metadata["vad_status"] = "pending" if _check_vad_available() else "unavailable"

    return jsonify(metadata)


# --- Chunked generation SSE progress ---
@app.route("/api/generate-progress/<job_id>")
def generate_progress(job_id):
    with generation_jobs_lock:
        job = generation_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job ID"}), 404

    def stream():
        q = job["queue"]
        while True:
            try:
                event = q.get(timeout=120)
            except Exception:
                yield f"data: {json.dumps({'phase': 'error', 'message': 'Timeout waiting for generation'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("phase") in ("done", "error"):
                break

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- List audio files ---
@app.route("/api/audio")
def list_audio():
    files = []
    if not os.path.exists(AUDIO_DIR):
        return jsonify(files)
    for fname in sorted(os.listdir(AUDIO_DIR), reverse=True):
        if fname.endswith(".json") and os.path.isfile(os.path.join(AUDIO_DIR, fname)):
            try:
                with open(os.path.join(AUDIO_DIR, fname), "r") as f:
                    files.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass  # skip partially-written or corrupt files
    return jsonify(files)


# --- Delete audio file (move to TRASH) ---
@app.route("/api/audio/<filename>", methods=["DELETE"])
def delete_audio(filename):
    basename = filename.rsplit(".", 1)[0]
    moved = False
    # Move all related files (wav, json, enhanced, cleaned, mp3 variants)
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(basename) and os.path.isfile(os.path.join(AUDIO_DIR, f)):
            shutil.move(os.path.join(AUDIO_DIR, f), os.path.join(TRASH_DIR, f))
            moved = True
    if moved:
        return jsonify({"status": "deleted", "filename": filename})
    return jsonify({"error": "File not found"}), 404


# --- Delete all audio files (move to TRASH) ---
@app.route("/api/audio", methods=["DELETE"])
def delete_all_audio():
    count = 0
    for f in os.listdir(AUDIO_DIR):
        fp = os.path.join(AUDIO_DIR, f)
        if os.path.isfile(fp):
            shutil.move(fp, os.path.join(TRASH_DIR, f))
            count += 1
    return jsonify({"status": "deleted", "count": count})


# --- Word alignment for karaoke ---
@app.route("/api/audio/<filename>/alignment")
def get_alignment(filename):
    """Return alignment data, triggering retroactive alignment for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(AUDIO_DIR, basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_alignment_available():
        return jsonify({"status": "unavailable"})

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"status": "aligning"})  # file being written, try again later

    status = metadata.get("alignment_status")

    if status == "ready":
        return jsonify({
            "status": "ready",
            "word_alignment": metadata.get("word_alignment", []),
        })

    if status == "failed":
        # Retry — previous failure may have been due to a fixable issue
        _start_alignment(basename)
        return jsonify({"status": "aligning"})

    if status == "aligning":
        # Check if thread is actually still running (may have crashed)
        with alignment_tasks_lock:
            if basename not in alignment_tasks:
                _start_alignment(basename)
        return jsonify({"status": "aligning"})

    # No alignment attempted yet (old file) — trigger retroactive alignment
    _start_alignment(basename)
    return jsonify({"status": "aligning"})


# --- Audio enhancement status ---
@app.route("/api/audio/<filename>/enhance-status")
def get_enhance_status(filename):
    """Return enhancement status, triggering retroactive enhancement for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(AUDIO_DIR, basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_enhance_available():
        return jsonify({"status": "unavailable"})

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"status": "enhancing"})  # file being written, try again later

    status = metadata.get("enhance_status")

    if status == "ready" and metadata.get("enhanced_filename"):
        enhanced_path = os.path.join(AUDIO_DIR, metadata["enhanced_filename"])
        if os.path.exists(enhanced_path):
            return jsonify({
                "status": "ready",
                "enhanced_filename": metadata["enhanced_filename"],
            })
        # File missing — re-enhance
        _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    if status == "failed":
        _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    if status == "enhancing":
        with enhance_tasks_lock:
            if basename not in enhance_tasks:
                _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    # No enhancement attempted yet (old file) — trigger retroactive
    _start_enhancement(basename)
    return jsonify({"status": "enhancing"})


# --- Silence removal (VAD) status ---
@app.route("/api/audio/<filename>/vad-status")
def get_vad_status(filename):
    """Return silence removal status, triggering retroactive cleaning for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(AUDIO_DIR, basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_vad_available():
        return jsonify({"status": "unavailable"})

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"status": "cleaning"})  # file being written, try again later

    status = metadata.get("vad_status")

    if status == "ready" and metadata.get("cleaned_filename"):
        cleaned_path = os.path.join(AUDIO_DIR, metadata["cleaned_filename"])
        if os.path.exists(cleaned_path):
            return jsonify({
                "status": "ready",
                "cleaned_filename": metadata["cleaned_filename"],
            })
        _start_vad(basename)
        return jsonify({"status": "cleaning"})

    if status == "failed":
        _start_vad(basename)
        return jsonify({"status": "cleaning"})

    if status == "normalizing":
        return jsonify({"status": "normalizing"})

    if status == "cleaning":
        with vad_tasks_lock:
            if basename not in vad_tasks:
                _start_vad(basename)
        return jsonify({"status": "cleaning"})

    # No VAD attempted yet — trigger retroactive
    _start_vad(basename)
    return jsonify({"status": "cleaning"})


# --- Locate ffmpeg helper ---
def _find_ffmpeg():
    bin_dir = os.path.join(os.path.dirname(__file__), "bin")
    local = os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return local if os.path.isfile(local) else shutil.which("ffmpeg")


ALIGNMENT_VERSION = 2  # Bump when alignment logic changes to invalidate cached data

# --- Alignment helpers (stable-ts) ---

def _check_alignment_available():
    """Check if stable-ts is importable. Cached after first call."""
    global alignment_available
    if alignment_available is not None:
        return alignment_available
    try:
        import stable_whisper  # noqa: F401
        alignment_available = True
    except ImportError:
        alignment_available = False
    return alignment_available


def _load_alignment_model():
    """Load (or return cached) stable-ts Whisper tiny.en model."""
    global alignment_model
    if alignment_model is not None:
        return alignment_model
    import stable_whisper
    with alignment_lock:
        if alignment_model is None:
            alignment_model = stable_whisper.load_model("tiny.en")
    return alignment_model


def _run_alignment(wav_path, prompt_text):
    """Run forced alignment. Returns list of {word, begin, end} or None."""
    try:
        model = _load_alignment_model()
        # Load audio as numpy array to avoid stable-ts needing ffmpeg in PATH
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Whisper expects 16kHz — resample if needed (KittenTTS outputs 24kHz)
        if sr != 16000:
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        result = model.align(audio, prompt_text, language="en", fast_mode=True)
        alignment = []
        for w in result.all_words():
            word_text = w.word.strip()
            if word_text:
                alignment.append({
                    "word": word_text,
                    "begin": round(w.start, 3),
                    "end": round(w.end, 3),
                })
        return alignment if alignment else None
    except Exception as e:
        print(f"[alignment] Error aligning {wav_path}: {e}")
        return None


def _audio_hash(wav_path):
    """Compute SHA-256 hash of audio file for cache validation."""
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _background_align(basename):
    """Run alignment in background thread, update metadata JSON when done."""
    json_path = os.path.join(AUDIO_DIR, basename + ".json")
    wav_path = os.path.join(AUDIO_DIR, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Check if alignment is cached, audio hasn't changed, and version matches
        current_hash = _audio_hash(wav_path)
        if (metadata.get("alignment_status") == "ready"
                and metadata.get("audio_hash") == current_hash
                and metadata.get("alignment_version") == ALIGNMENT_VERSION
                and metadata.get("word_alignment")):
            return

        metadata["alignment_status"] = "aligning"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        prompt_text = metadata.get("prompt", "")
        if not prompt_text.strip():
            metadata["alignment_status"] = "failed"
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return

        alignment = _run_alignment(wav_path, prompt_text)

        if alignment:
            metadata["alignment_status"] = "ready"
            metadata["word_alignment"] = alignment
            metadata["audio_hash"] = current_hash
            metadata["alignment_version"] = ALIGNMENT_VERSION
        else:
            metadata["alignment_status"] = "failed"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"[alignment] Background align failed for {basename}: {e}")
    finally:
        with alignment_tasks_lock:
            alignment_tasks.pop(basename, None)


def _start_alignment(basename):
    """Spawn alignment thread if not already running for this file."""
    if not _check_alignment_available():
        return
    with alignment_tasks_lock:
        if basename in alignment_tasks:
            return
        t = threading.Thread(target=_background_align, args=(basename,), daemon=True)
        alignment_tasks[basename] = t
        t.start()


# --- Enhancement helpers (LavaSR) ---

def _check_enhance_available():
    """Check if LavaSR is importable. Cached after first call."""
    global enhance_available
    if enhance_available is not None:
        return enhance_available
    try:
        from LavaSR.model import LavaEnhance  # noqa: F401
        enhance_available = True
    except ImportError:
        enhance_available = False
    return enhance_available


def _load_enhance_model():
    """Load (or return cached) LavaSR enhancement model."""
    global enhance_model
    if enhance_model is not None:
        return enhance_model
    from LavaSR.model import LavaEnhance
    with enhance_lock:
        if enhance_model is None:
            enhance_model = LavaEnhance("YatharthS/LavaSR", "cpu")
    return enhance_model


def _run_enhance(wav_path):
    """Enhance audio file. Returns enhanced filename or None."""
    try:
        model = _load_enhance_model()
        audio, sr = model.load_audio(wav_path)
        enhanced = model.enhance(audio)
        enhanced_np = enhanced.cpu().numpy().squeeze()

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        enhanced_name = f"{basename}_enhanced.wav"
        enhanced_path = os.path.join(AUDIO_DIR, enhanced_name)
        sf.write(enhanced_path, enhanced_np, 48000)
        return enhanced_name
    except Exception as e:
        print(f"[enhance] Error enhancing {wav_path}: {e}")
        return None


def _background_enhance(basename):
    """Run enhancement in background thread, update metadata JSON when done.
    Automatically chains into silence removal (VAD) when enhancement completes."""
    json_path = os.path.join(AUDIO_DIR, basename + ".json")
    wav_path = os.path.join(AUDIO_DIR, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Skip if already enhanced and file exists
        if (metadata.get("enhance_status") == "ready"
                and metadata.get("enhanced_filename")
                and os.path.exists(os.path.join(AUDIO_DIR, metadata["enhanced_filename"]))):
            # Still chain VAD if not done yet
            if metadata.get("vad_status") not in ("ready", "cleaning"):
                _start_vad(basename)
            return

        metadata["enhance_status"] = "enhancing"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        enhanced_name = _run_enhance(wav_path)

        if enhanced_name:
            metadata["enhance_status"] = "ready"
            metadata["enhanced_filename"] = enhanced_name
        else:
            metadata["enhance_status"] = "failed"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Chain: start silence removal after enhancement completes
        _start_vad(basename)

    except Exception as e:
        print(f"[enhance] Background enhance failed for {basename}: {e}")
        # Still try VAD even if enhancement failed
        _start_vad(basename)
    finally:
        with enhance_tasks_lock:
            enhance_tasks.pop(basename, None)


def _start_enhancement(basename):
    """Spawn enhancement thread if not already running for this file."""
    if not _check_enhance_available():
        return
    with enhance_tasks_lock:
        if basename in enhance_tasks:
            return
        t = threading.Thread(target=_background_enhance, args=(basename,), daemon=True)
        enhance_tasks[basename] = t
        t.start()


# --- Silence removal helpers (Silero VAD) ---

def _check_vad_available():
    """Check if torch is importable (Silero VAD needs it). Cached after first call."""
    global vad_available
    if vad_available is not None:
        return vad_available
    try:
        import torch  # noqa: F401
        vad_available = True
    except ImportError:
        vad_available = False
    return vad_available


def _load_vad_model():
    """Load (or return cached) Silero VAD model."""
    global vad_model, vad_utils
    if vad_model is not None:
        return vad_model, vad_utils
    import torch
    with vad_lock:
        if vad_model is None:
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
            vad_model = model
            vad_utils = utils
    return vad_model, vad_utils


def _run_silence_removal(wav_path, max_silence_ms=500):
    """Remove silences longer than max_silence_ms using Silero VAD.
    Short pauses (<= threshold) are kept intact for natural speech."""
    try:
        import torch
        model, utils = _load_vad_model()
        get_speech_timestamps = utils[0]

        # Read audio with soundfile and resample to 16kHz (avoids torchaudio dependency)
        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        orig_audio = audio_np.copy()
        orig_sr = sr
        if sr != 16000:
            target_len = int(len(audio_np) * 16000 / sr)
            audio_np = np.interp(
                np.linspace(0, len(audio_np), target_len, endpoint=False),
                np.arange(len(audio_np)),
                audio_np,
            ).astype(np.float32)
        wav_16k = torch.from_numpy(audio_np)
        timestamps = get_speech_timestamps(
            wav_16k, model,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        if not timestamps:
            return None

        # Map 16kHz sample indices to original sample rate
        ratio = orig_sr / 16000

        # Merge segments, keeping silences <= max_silence_ms intact
        chunks = []
        for i, seg in enumerate(timestamps):
            start = int(seg["start"] * ratio)
            end = int(seg["end"] * ratio)

            if i > 0:
                prev_end = int(timestamps[i - 1]["end"] * ratio)
                gap_ms = ((start - prev_end) / orig_sr) * 1000

                if gap_ms <= max_silence_ms:
                    # Keep the silence — include gap + speech
                    chunks.append(orig_audio[prev_end:end])
                else:
                    # Drop the long silence, just add speech segment
                    chunks.append(orig_audio[start:end])
            else:
                chunks.append(orig_audio[start:end])

        if not chunks:
            return None

        cleaned = np.concatenate(chunks)
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        cleaned_name = f"{basename}_cleaned.wav"
        cleaned_path = os.path.join(AUDIO_DIR, cleaned_name)
        sf.write(cleaned_path, cleaned, orig_sr)
        return cleaned_name
    except Exception as e:
        print(f"[vad] Error removing silence from {wav_path}: {e}")
        return None


def _run_loudnorm(wav_path):
    """Normalize audio volume using ffmpeg loudnorm. Overwrites the file in-place."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False
    tmp_path = wav_path + ".tmp.wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-i", wav_path,
             "-af", "loudnorm=I=-16:LRA=11:TP=-1.5,aresample=24000",
             tmp_path],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, wav_path)
            return True
        else:
            print(f"[loudnorm] ffmpeg failed: {result.stderr.decode(errors='replace')[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[loudnorm] Timed out normalizing {wav_path}")
        return False
    except Exception as e:
        print(f"[loudnorm] Error normalizing {wav_path}: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _background_vad(basename, max_silence_ms=500):
    """Run silence removal in background thread, update metadata JSON when done.
    Also runs loudnorm on the cleaned file if ffmpeg is available."""
    json_path = os.path.join(AUDIO_DIR, basename + ".json")
    # Prefer enhanced audio if available, otherwise use original
    with open(json_path, "r") as f:
        metadata = json.load(f)

    enhanced_name = metadata.get("enhanced_filename")
    if enhanced_name and os.path.exists(os.path.join(AUDIO_DIR, enhanced_name)):
        wav_path = os.path.join(AUDIO_DIR, enhanced_name)
    else:
        wav_path = os.path.join(AUDIO_DIR, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    # Read max_silence_ms from metadata if stored (from generate request)
    max_silence_ms = metadata.get("max_silence_ms", max_silence_ms)

    try:
        # Skip if already cleaned and file exists
        if (metadata.get("vad_status") == "ready"
                and metadata.get("cleaned_filename")
                and os.path.exists(os.path.join(AUDIO_DIR, metadata["cleaned_filename"]))):
            return

        metadata["vad_status"] = "cleaning"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cleaned_name = _run_silence_removal(wav_path, max_silence_ms=max_silence_ms)

        if cleaned_name:
            # Run loudnorm on the cleaned file
            metadata["vad_status"] = "normalizing"
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            cleaned_path = os.path.join(AUDIO_DIR, cleaned_name)
            if _run_loudnorm(cleaned_path):
                print(f"[loudnorm] Normalized {cleaned_name}")
            else:
                print(f"[loudnorm] Skipped (ffmpeg unavailable or failed) for {cleaned_name}")

            metadata["vad_status"] = "ready"
            metadata["cleaned_filename"] = cleaned_name
        else:
            metadata["vad_status"] = "failed"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"[vad] Background silence removal failed for {basename}: {e}")
        # Always finalize metadata so status doesn't stay stuck
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            metadata["vad_status"] = "failed"
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass
    finally:
        with vad_tasks_lock:
            vad_tasks.pop(basename, None)


def _start_vad(basename, max_silence_ms=500):
    """Spawn silence removal thread if not already running for this file."""
    if not _check_vad_available():
        return
    with vad_tasks_lock:
        if basename in vad_tasks:
            return
        t = threading.Thread(target=_background_vad, args=(basename, max_silence_ms), daemon=True)
        vad_tasks[basename] = t
        t.start()


# --- Check if MP3 exists (returns 200 always, avoids 404 console noise) ---
@app.route("/api/audio/<filename>/mp3-check")
def check_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"exists": False})
    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_name)
    return jsonify({"exists": os.path.exists(mp3_path)})


# --- Serve cached MP3 ---
@app.route("/api/audio/<filename>/mp3")
def serve_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_name)
    if not os.path.exists(mp3_path):
        return jsonify({"error": "MP3 not found — convert first"}), 404
    return send_from_directory(AUDIO_DIR, mp3_name, as_attachment=True)


# --- Convert WAV to MP3 with SSE progress ---
@app.route("/api/audio/<filename>/mp3-convert")
def convert_to_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    wav_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(wav_path):
        return jsonify({"error": "File not found"}), 404

    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_name)

    # Already converted — instant done
    if os.path.exists(mp3_path):
        def _done():
            yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
        return Response(
            _done(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return jsonify({"error": "ffmpeg not found. Place ffmpeg in bin/ or install it system-wide."}), 501

    # Get total duration for progress calculation
    total_duration = 0.0
    json_path = wav_path.rsplit(".", 1)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            total_duration = json.load(f).get("duration_seconds", 0.0)
    if total_duration <= 0:
        try:
            info = sf.info(wav_path)
            total_duration = info.duration
        except Exception:
            pass

    def stream():
        yield f"data: {json.dumps({'phase': 'converting', 'progress': 0})}\n\n"

        proc = subprocess.Popen(
            [ffmpeg, "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2",
             "-progress", "pipe:1", "-nostats", "-y", mp3_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        last_pct = 0
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    if total_duration > 0:
                        pct = min(99, int((us / 1_000_000) / total_duration * 100))
                        if pct > last_pct:
                            last_pct = pct
                            yield f"data: {json.dumps({'phase': 'converting', 'progress': pct})}\n\n"
                except (ValueError, ZeroDivisionError):
                    pass
            elif line == "progress=end":
                break

        proc.wait(timeout=30)

        if proc.returncode == 0:
            yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
        else:
            err = proc.stderr.read()[:200] if proc.stderr else "Unknown error"
            yield f"data: {json.dumps({'phase': 'error', 'message': err})}\n\n"

    return Response(
        stream(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# --- Serve audio files ---
@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KittenTTS Studio Backend")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on")
    args = parser.parse_args()

    port = args.port if args.port else find_available_port(5000)
    print(f"\n>>> KittenTTS Studio running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
