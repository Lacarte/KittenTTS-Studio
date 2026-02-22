# KittenTTS-Studio — Project Status & Session Log

**Repository:** https://github.com/Lacarte/KittenTTS-Studio
**Branch:** main

## Current State (2026-02-21)

### Latest Commit
- `1794108` — `feat: initial KittenTTS Web Studio application` (10 files, 4,541 lines)

---

## Architecture Overview

### Processing Pipeline (per generation)
```
User clicks Generate
  │
  ├─ Step 1: Generate audio (KittenTTS) → saves WAV + JSON
  │    ├─ Start alignment thread (stable-ts, parallel)
  │    └─ Start enhancement thread (LavaSR)
  │
  ├─ Step 2: Enhancement (LavaSR) → saves _enhanced.wav (48kHz)
  │    └─ Chains into VAD when done
  │
  ├─ Step 3: Silence Removal (Silero VAD) → saves _cleaned.wav
  │    └─ Uses enhanced audio if available, preserves gaps ≤ max_silence_ms
  │
  ├─ Step 4: Loudnorm (ffmpeg) → overwrites _cleaned.wav in-place
  │    └─ Runs inside VAD background thread after silence removal
  │
  └─ Step 5: MP3 Conversion (ffmpeg, frontend-driven) → saves _cleaned.mp3
```

### Audio File Variants
```
generated_assets/
  tts/
    some-text_20260221_143052.wav          # Original (24kHz)
    some-text_20260221_143052.json         # Metadata
    some-text_20260221_143052_enhanced.wav # Enhanced (48kHz, LavaSR)
    some-text_20260221_143052_cleaned.wav  # Silence removed + loudnorm
    some-text_20260221_143052_cleaned.mp3  # MP3 of cleaned version
    TRASH/                                 # Soft-deleted TTS files
  force-alignment/
    audio-name_20260222_alignment.json     # Standalone alignment results
    TRASH/                                 # Soft-deleted alignment files
```

### Player Version Selector
3-button group in player footer: **Original** / **Enhanced** / **Cleaned**
- Preserves playback position when switching
- Polls backend for readiness, enables buttons as versions become available

---

## Completed Features (Committed)

| Commit | Feature |
|--------|---------|
| `e6b94c4` | Initial web app: Flask backend, single-file frontend, model download SSE, generation, history |
| `1a2bd1b` | MP3 conversion with live progress, project README |
| `9399c00` | History render bug fix, setup scripts, UI improvements |
| `fddd7b5` | Karaoke word highlighting with stable-ts forced alignment |
| `e1c349c` | MP3 icon and alignment button animations |
| `bec414f` | Alignment icon swap to waveform, player separator |
| `1ea0952` | LavaSR audio enhancement, speed control (0.5x–2.0x), UI polish |
| `b29bece` | Silero VAD silence removal, 4-step stepper, version selector, clean_for_tts, silence threshold (0.2s–1.0s) |
| `cdaeee5` | Soft-delete with TRASH folder, delete-all history button |
| `1794108` | Initial commit on new repo (KittenTTS-Studio) — full app with all features above |

---

## Key Backend Components (`backend.py`)

### Globals & Models
- `alignment_model`, `alignment_lock` — stable-ts/whisper (lazy-loaded)
- `enhance_model`, `enhance_lock` — LavaSR (lazy-loaded)
- `vad_model`, `vad_utils`, `vad_lock` — Silero VAD (lazy-loaded via torch.hub)
- Each has `*_tasks` dict + `*_tasks_lock` for tracking background threads
- Each has `_check_*_available()` with cached result

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Status + feature flags (ffmpeg, alignment, enhance, vad) |
| `/api/models` | GET | List 5 models with specs |
| `/api/voices` | GET | 8 voices |
| `/api/model-status/<id>` | GET | Check if model is cached |
| `/api/download-model/<id>` | GET (SSE) | Stream download progress |
| `/api/generate` | POST | Generate audio (accepts model, voice, prompt, speed, max_silence_ms) |
| `/api/normalize` | POST | Normalize text for TTS |
| `/api/generation` | GET | List history |
| `/api/generation` | DELETE | Delete all (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single (move to TRASH) |
| `/generation/<file>` | GET | Serve audio file |
| `/api/generation/<file>/alignment` | GET | Word alignment data |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-convert` | GET (SSE) | Convert WAV to MP3 with progress |
| `/api/generation/<file>/mp3` | GET | Serve cached MP3 |
| `/api/generation/alignments` | GET | List alignment data (TTS + standalone) |
| `/api/force-align` | POST | Standalone force alignment (audio + text upload) |

### Text Processing
- `clean_for_tts()` — strips markdown `*_#\`~`, replaces URLs with "link", collapses whitespace (applied before TTS generation)
- `normalize_for_tts()` — full pipeline: symbols, contractions, abbreviations, currency, units, dates, time, ordinals, numbers (called via `/api/normalize` endpoint, user-triggered via Format button)

---

## Key Frontend Components (`frontend/index.html`)

### STATE Object
```javascript
{
  selectedModel, selectedVoice, history, nowPlaying,
  isGenerating, darkMode, downloadEventSource, eqInterval,
  ffmpeg, alignment, alignmentAvailable, alignmentPollTimer, activeWordIndex,
  enhanceAvailable, enhancePollTimer,
  vadAvailable, vadPollTimer,
  activeVersion,      // 'original' | 'enhanced' | 'cleaned'
  processingStep,     // 0=idle, 1–5=pipeline steps
}
```

### UI Sections
1. **Header** — logo, dark mode toggle, hamburger menu
2. **Model selector** — dropdown with 5 models + size badge
3. **Voice grid** — 8 clickable voice chips
4. **Speed + Silence selectors** — dropdowns (0.5x–2.0x, 0.2s–1.0s)
5. **Prompt** — textarea with word/token count + Format button
6. **Processing stepper** — 5-step visual pipeline (shown during generation)
7. **Generate button** — changes color per step (coral→teal→gold→orange→purple)
8. **History** — scrollable list with play, text expand, alignment buttons + delete-all
9. **Fixed footer player** — seek bar, play/pause, karaoke text, version selector, cleaned MP3 download, delete

### Key JS Functions
- `handleGenerate()` — orchestrates 5-step pipeline with sequential polling
- `pollUntilDone(url, done, pending, interval, onProgress)` — generic status poller
- `switchAudioVersion(ver)` — swap player src, preserve position
- `pollVersionStatuses(filename)` — parallel polling for enhance + VAD readiness
- `normalizePrompt()` — calls `/api/normalize`, replaces prompt text
- `autoConvertMp3(wavFilename)` — SSE-driven MP3 conversion
- `downloadCleanedMp3()` — fetch + blob download of cleaned MP3

---

## Dependencies (`requirements.txt`)
```
kittentts-0.8.0 (wheel)
flask, flask-cors
openai-whisper, stable-ts     # alignment
git+LavaSR                     # enhancement
num2words                      # text normalization
```
Implicit: torch, numpy, soundfile, huggingface-hub, onnxruntime, spacy (pulled by kittentts/whisper/LavaSR)

### Optional System Dependencies
- **ffmpeg** — MP3 conversion, loudnorm. Place in `bin/` or install system-wide.
- **eSpeak-ng** — not used (switched from aeneas to stable-ts for alignment)

---

## Upcoming: Prosody (Expression) in Audio via Parselmouth

**Status: NOT STARTED**

KittenTTS generates flat, neutral speech. Post-generation expression manipulation (pitch, expressiveness, rate) via **Parselmouth** (Python wrapper for Praat) — lightweight (~10MB), no GPU, parametric slider controls, sub-100ms processing.

### Pipeline Position
```
Generate (24kHz) → Expression (Parselmouth, in-place) → Loudnorm → Enhancement → VAD → MP3
```
Applied synchronously during generation (before WAV write). The expression-adjusted audio becomes the base file.

### Changes Required

**`requirements.txt`** — Add `parselmouth`

**`backend.py`**:
- Add `expression_available = None` global
- Add `_check_expression_available()` — lazy import, cached result
- Add `_run_expression(audio_np, sample_rate, pitch_shift, pitch_range, rate_factor)`:
  - No-op fast path if all defaults (0, 1.0, 1.0)
  - Parselmouth PSOLA: `To Manipulation` → pitch tier → `Multiply frequencies` → `Get resynthesis (overlap-add)`
  - Duration tier for rate adjustment
- Update `/api/health` — add `"expression": _check_expression_available()`
- Update `/api/generate` — accept `pitch_shift` (-6 to +6), `pitch_range` (0.5 to 2.0), `rate_adjust` (0.8 to 1.3)
- Integrate into single-sentence and chunked generation paths

**`frontend/index.html`**:
- Add `STATE.expressionAvailable`, set from health response
- Settings page — "Expression" card with 3 sliders (Pitch Shift, Pitch Range, Rate Adjust)
- Pass expression params in `handleGenerate()` request body
- Show expression badge in history when non-default

---

## Known Issues / TODO
- [x] ~~Commit pending changes (normalize + loudnorm + bug fixes)~~ — included in initial commit
- [x] ~~Project renamed to **KittenTTS-Studio**, pushed to GitHub~~
- [ ] Prosody (Expression) in Audio — see section above
- [ ] Long text chunking: split by sentence for better prosody on long prompts
- [ ] Batch generation queue for automation pipelines
- [ ] Waveform visualization during playback
- [ ] Dark mode persistence improvements
- [ ] Favicon (currently 404)
