# KittenTTS-Studio

A web studio for [KittenTTS](https://github.com/KittenML/KittenTTS) — ultra-lightweight open-source text-to-speech. Generate realistic speech from text in your browser, no GPU required.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![KittenTTS 0.8](https://img.shields.io/badge/kittentts-0.8.0-coral)
![Flask](https://img.shields.io/badge/flask-backend-teal)

## Features

- **Browser-based UI** — single-page frontend served by Flask, no build step
- **5 models** — Mini (80M), Micro (40M), Nano, Nano INT8, Nano FP32
- **8 voices** — Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- **Real-time download progress** — SSE streams per-file progress when fetching models from HuggingFace Hub
- **Audio enhancement** — LavaSR upscaling from 24kHz to 48kHz
- **Silence removal** — Silero VAD trims dead air with configurable threshold (0.2s–1.0s)
- **Loudness normalization** — ffmpeg loudnorm for consistent volume
- **Karaoke word highlighting** — stable-ts forced alignment with real-time word tracking
- **Text normalization** — expands numbers, currency, abbreviations, dates, and symbols for cleaner speech
- **Speed control** — adjustable playback from 0.5x to 2.0x
- **MP3 export** — one-click WAV to MP3 conversion with live progress
- **Audio history** — generated files saved with metadata, replayable from the UI
- **Soft delete** — files moved to TRASH folder, with delete-all option
- **Version selector** — switch between Original, Enhanced, and Cleaned audio in the player
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
- **ffmpeg** (optional, for MP3 conversion and loudnorm) — place `ffmpeg.exe` in `bin/` or install system-wide

Python dependencies:

```
kittentts==0.8.0   (pulls numpy, soundfile, onnxruntime, spacy, torch, etc.)
flask, flask-cors
openai-whisper, stable-ts   (forced alignment / karaoke)
LavaSR                      (audio enhancement)
num2words                   (text normalization)
```

## Processing Pipeline

```
User clicks Generate
  │
  ├─ Step 1: Generate audio (KittenTTS) → WAV + JSON metadata
  ├─ Step 2: Enhance (LavaSR) → 48kHz upscaled WAV
  ├─ Step 3: Remove silence (Silero VAD) → cleaned WAV
  ├─ Step 4: Normalize loudness (ffmpeg loudnorm) → consistent volume
  └─ Step 5: Convert to MP3 (ffmpeg) → final MP3
```

## Project Structure

```
KittenTTS-Studio/
├── main.py             # Original CLI script
├── backend.py          # Flask API server
├── requirements.txt    # Python dependencies
├── setup.bat           # Environment setup
├── runner.bat          # One-click launcher
├── bin/                # Local ffmpeg (optional, gitignored)
├── generated_assets/   # All generated output (gitignored)
│   ├── tts/            # TTS audio + metadata
│   │   └── TRASH/      # Soft-deleted TTS files
│   └── force-alignment/ # Standalone alignment results
│       └── TRASH/      # Soft-deleted alignment files
└── frontend/
    └── index.html      # Single-file UI (inline CSS/JS, Tailwind CDN)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Server status + feature flags |
| `/api/models` | GET | List available models |
| `/api/voices` | GET | List available voices |
| `/api/model-status/<id>` | GET | Check if model is cached |
| `/api/download-model/<id>` | GET | SSE — download model with progress |
| `/api/generate` | POST | Generate audio from `{model, voice, prompt, speed, max_silence_ms}` |
| `/api/normalize` | POST | Normalize text for TTS |
| `/api/generation` | GET | List all generated audio metadata |
| `/api/generation` | DELETE | Delete all files (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single file (move to TRASH) |
| `/api/generation/<file>/alignment` | GET | Word alignment data |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-convert` | GET | SSE — convert WAV to MP3 |
| `/api/generation/<file>/mp3` | GET | Serve converted MP3 |
| `/api/generation/alignments` | GET | List alignment data (TTS + standalone) |
| `/api/force-align` | POST | Standalone force alignment (audio + text upload) |
| `/generation/<file>` | GET | Serve audio file |

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
