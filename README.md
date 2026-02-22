# KittenTTS Studio

A web studio for [KittenTTS](https://github.com/KittenML/KittenTTS) -- ultra-lightweight open-source text-to-speech. Generate realistic speech from text in your browser, no GPU required.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![KittenTTS 0.8](https://img.shields.io/badge/kittentts-0.8.0-coral)
![Flask](https://img.shields.io/badge/flask-backend-teal)

## Features

- **Browser-based UI** -- dark-themed single-page app served by Flask, no build step
- **5 models** -- Mini (80M), Micro (40M), Nano, Nano INT8, Nano FP32
- **8 voices** -- Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- **Real-time download progress** -- SSE streams per-file progress when fetching models from HuggingFace Hub
- **Breathing-block chunking** -- long texts split into 150-200 char blocks with merge logic for natural pacing
- **Async generation with abort** -- chunked generation runs in background threads; cancel anytime
- **Audio enhancement** -- LavaSR upscaling from 24kHz to 48kHz
- **Silence removal** -- Silero VAD trims dead air with configurable threshold (0.2s-1.0s)
- **Loudness normalization** -- ffmpeg loudnorm for consistent volume
- **Karaoke word highlighting** -- stable-ts forced alignment with real-time word tracking and click-to-seek
- **Text normalization** -- expands numbers, currency, abbreviations, dates, and symbols; auto-formats into breathing blocks
- **Speed control** -- adjustable playback from 0.5x to 2.0x (persisted)
- **MP3 export** -- WAV to MP3 conversion with live SSE progress
- **3-version player** -- switch between Original, Enhanced, and Cleaned audio; preserves seek position
- **Standalone force alignment** -- upload any audio + transcript for word-level timestamps
- **Unified library** -- TTS and alignment history merged with type tags and filter tabs
- **Per-generation subfolders** -- each job saved in its own directory under `generated_assets/tts/`
- **Soft delete** -- files moved to TRASH folder, with delete-all option and confirmation modal
- **Sidebar navigation** -- collapsible sidebar with 4 pages (TTS, Alignment, Library, Settings)
- **Dark theme** -- always-dark UI with navy/teal/coral palette
- **Keyboard shortcut** -- Ctrl+Enter to generate
- **Open folder** -- reveal generation output in OS file explorer
- **One-click startup** -- `runner.bat` finds a free port, starts the server, opens the browser

## Quick Start (Windows)

### 1. Setup

```
setup.bat
```

Creates a Python 3.12 venv and installs dependencies. If Python 3.12 isn't found on your system, the script downloads and installs it locally.

### 2. Run

```
runner.bat
```

Starts the backend on an available port (default 5000), waits for the health check to pass, then opens your browser.

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
- **ffmpeg** (optional, for MP3 conversion and loudnorm) -- place `ffmpeg.exe` in `bin/` or install system-wide

Python dependencies:

```
kittentts==0.8.0   (pulls numpy, soundfile, onnxruntime, spacy, torch, etc.)
flask, flask-cors
loguru                 (structured logging with rotation)
openai-whisper, stable-ts   (forced alignment / karaoke)
LavaSR                      (audio enhancement)
num2words                   (text normalization)
```

## Processing Pipeline

```
User clicks Generate
  |
  +-- Step 1: Generate audio (KittenTTS) -> WAV + JSON metadata
  |     +-- Start alignment thread (stable-ts, parallel)
  |     +-- Start enhancement thread (LavaSR)
  |
  +-- Step 2: Enhancement (LavaSR) -> 48kHz upscaled WAV
  |     +-- Chains into VAD when done
  |
  +-- Step 3: Silence Removal (Silero VAD) -> cleaned WAV
  |     +-- Uses enhanced audio if available, preserves gaps <= max_silence_ms
  |
  +-- Step 4: Loudnorm (ffmpeg) -> overwrites cleaned WAV in-place
  |     +-- Runs inside VAD background thread after silence removal
  |
  +-- Step 5: MP3 Conversion (ffmpeg, frontend-driven) -> final MP3
```

For long texts, the breathing-block chunker splits into 150-200 char blocks before generation, then crossfade-concatenates the resulting audio chunks.

## Project Structure

```
KittenTTS-Studio/
+-- main.py             # Original CLI script (unchanged)
+-- backend.py          # Flask API server (~2,100 lines)
+-- requirements.txt    # Python dependencies
+-- setup.bat           # Environment setup (auto-downloads Python 3.12 if needed)
+-- runner.bat          # One-click launcher with health-check polling
+-- CLAUDE.md           # AI assistant project brief
+-- PLAN.md             # Architecture notes and session log
+-- bin/                # Local ffmpeg (optional, gitignored)
+-- logs/               # Loguru rotating logs (gitignored)
+-- generated_assets/   # All generated output (gitignored)
|   +-- tts/            # Per-generation subfolders
|   |   +-- some-text_20260221_143052/
|   |   |   +-- some-text_20260221_143052.wav          # Original (24kHz)
|   |   |   +-- some-text_20260221_143052.json         # Metadata
|   |   |   +-- some-text_20260221_143052_enhanced.wav  # Enhanced (48kHz)
|   |   |   +-- some-text_20260221_143052_cleaned.wav   # Silence removed + loudnorm
|   |   |   +-- some-text_20260221_143052_cleaned.mp3   # MP3 of cleaned
|   |   +-- TRASH/      # Soft-deleted TTS files
|   +-- force-alignment/ # Standalone alignment results
|       +-- TRASH/       # Soft-deleted alignment files
+-- frontend/
    +-- index.html      # Single-file UI (~2,700 lines, inline CSS/JS, Tailwind CDN)
```

## UI Pages

| Page | Description |
|------|-------------|
| **TTS** | Model selector, voice grid, prompt editor with Format/Copy buttons, 4-step processing stepper, Generate button |
| **Alignment** | Standalone force alignment: drag-and-drop audio upload, transcript editor, karaoke playback |
| **Library** | Unified history with filter tabs (All / TTS / Alignment), play/delete/metadata per item |
| **Settings** | Speed control, silence threshold, feature availability status, About section |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Server status + feature flags (ffmpeg, alignment, enhance, VAD) |
| `/api/models` | GET | List available models |
| `/api/voices` | GET | List available voices |
| `/api/normalize` | POST | Normalize text for TTS (expand numbers, format breathing blocks) |
| `/api/model-status/<id>` | GET | Check if model is cached |
| `/api/download-model/<id>` | GET (SSE) | Download model with real-time progress |
| `/api/generate` | POST | Generate audio from `{model, voice, prompt, speed, max_silence_ms}` |
| `/api/generate-progress/<job_id>` | GET (SSE) | Stream chunked generation progress |
| `/api/generate-abort/<job_id>` | POST | Abort in-flight generation |
| `/api/generation` | GET | List all generated audio metadata |
| `/api/generation` | DELETE | Delete all files (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single file (move to TRASH) |
| `/api/generation/<file>/alignment` | GET | Word alignment data (triggers retroactively if needed) |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-convert` | GET (SSE) | Convert WAV to MP3 with progress |
| `/api/generation/<file>/mp3-check` | GET | Check if MP3 exists |
| `/api/generation/<file>/mp3` | GET | Serve converted MP3 |
| `/api/generation/alignments` | GET | List all alignment data (TTS + standalone) |
| `/api/generation/force-alignment` | GET | List standalone force-alignment results |
| `/api/generation/alignment/<folder>` | DELETE | Soft-delete alignment folder |
| `/api/force-align` | POST | Standalone force alignment (audio + text upload) |
| `/api/open-generation-folder` | POST | Open OS file explorer at generation folder |
| `/generation/<file>` | GET | Serve audio file |
| `/generation/force-alignment/<file>` | GET | Serve force-alignment audio |

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
