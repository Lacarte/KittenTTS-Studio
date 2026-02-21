# KittenTTS Studio

A web interface for [KittenTTS](https://github.com/KittenML/KittenTTS) — the ultra-lightweight open-source text-to-speech model. Generate realistic speech from text in your browser, no GPU required.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![KittenTTS 0.8](https://img.shields.io/badge/kittentts-0.8.0-coral)
![Flask](https://img.shields.io/badge/flask-backend-teal)

## Features

- **Browser-based UI** — single-page frontend served by Flask, no build step
- **5 models** — Mini (80M), Micro (40M), Nano, Nano INT8, Nano FP32
- **8 voices** — Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- **Real-time download progress** — SSE streams per-file progress when fetching models from HuggingFace Hub
- **Audio history** — generated files saved with metadata, replayable from the UI
- **MP3 export** — one-click WAV to MP3 conversion with live progress (requires ffmpeg)
- **Dark mode** — toggle with persistence
- **Keyboard shortcut** — Ctrl+Enter to generate
- **One-click startup** — `runner.bat` finds a free port, starts the server, opens the browser

## Quick Start (Windows)

### 1. Setup

```
setup.bat
```

Creates a Python 3.12 venv and installs dependencies.

### 2. Run

```
runner.bat
```

Starts the backend on an available port and opens your browser.

### Manual Start

```bash
# activate venv
venv\Scripts\activate

# run the server
python backend.py --port 5000
```

Then open `http://localhost:5000`.

## Requirements

- **Python 3.12**
- **ffmpeg** (optional, for MP3 conversion) — place `ffmpeg.exe` in `bin/` or install system-wide

Python dependencies are minimal:

```
kittentts==0.8.0   (pulls numpy, soundfile, onnxruntime, spacy, etc.)
flask
flask-cors
```

## Project Structure

```
kittenTTS/
├── main.py             # Original CLI script
├── backend.py          # Flask API server
├── requirements.txt    # Python dependencies
├── setup.bat           # Environment setup
├── runner.bat          # One-click launcher
├── bin/                # Local ffmpeg (optional, gitignored)
├── audio/              # Generated WAV/MP3 + JSON metadata (gitignored)
└── frontend/
    └── index.html      # Single-file UI (inline CSS/JS, Tailwind CDN)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Server status |
| `/api/models` | GET | List available models |
| `/api/voices` | GET | List available voices |
| `/api/model-status/<id>` | GET | Check if model is cached |
| `/api/download-model/<id>` | GET | SSE stream — download model with progress |
| `/api/generate` | POST | Generate audio from `{model, voice, prompt}` |
| `/api/audio` | GET | List all generated audio metadata |
| `/api/audio/<file>` | DELETE | Delete an audio file |
| `/api/audio/<file>/mp3-convert` | GET | SSE stream — convert WAV to MP3 with progress |
| `/api/audio/<file>/mp3` | GET | Download converted MP3 |
| `/audio/<file>` | GET | Serve audio file |

## Models

| ID | Name | Params | Size | Repository |
|----|------|--------|------|------------|
| `mini` | Kitten TTS Mini | 80M | 80MB | KittenML/kitten-tts-mini-0.8 |
| `micro` | Kitten TTS Micro | 40M | 41MB | KittenML/kitten-tts-micro-0.8 |
| `nano` | Kitten TTS Nano | 15M | 56MB | KittenML/kitten-tts-nano-0.8 |
| `nano-int8` | Kitten TTS Nano INT8 | 15M | 19MB | KittenML/kitten-tts-nano-0.8-int8 |
| `nano-fp32` | Kitten TTS Nano FP32 | 15M | ~56MB | KittenML/kitten-tts-nano-0.8-fp32 |

Models are downloaded from HuggingFace Hub on first use and cached locally.

## Credits

Built on top of [KittenTTS](https://github.com/KittenML/KittenTTS) by KittenML.
