Here is the **complete, refined prompt**:

---

## Complete Project Brief: KittenTTS Web Studio

### Goal
Create a complete web application for KittenTTS text-to-speech generation with a Flask backend, beautiful Flask-served frontend, real-time model download progress, audio generation, history management, and one-click startup.

---

### Project Structure
```
kittentts-studio/
├── main.py                 # Original CLI script (UNCHANGED)
├── backend.py              # Flask API server
├── requirements.txt        # Python dependencies
├── runner.bat              # Windows one-click launcher
├── setup.bat               # Environment setup
├── models/                 # Cached TTS models (persisted)
├── audio/                  # Generated audio + metadata
│   ├── reports-of-my-death_20250220_143052.wav
│   └── reports-of-my-death_20250220_143052.json
└── frontend/
    └── index.html          # Single-file, inline everything
```

---

### 1. BACKEND (`backend.py`)

**Framework:** Flask + Flask-CORS + Flask-SSE (or pure SSE via `yield`)

**Core Logic:** Copy from `main.py`:
```python
from kittentts import KittenTTS
import soundfile as sf
import time

# Model initialization (downloads from HF Hub on first run)
m = KittenTTS("KittenML/kitten-tts-mini-0.8")

# Generation with timing
start = time.perf_counter()
audio = m.generate(prompt, voice=voice)
end = time.perf_counter()

# Metrics
duration_generated = len(audio) / 24000
inference_time = end - start
rtf = inference_time / duration_generated
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status + current port |
| `/api/models` | GET | List 5 models with specs |
| `/api/voices` | GET | 8 voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo |
| `/api/model-status/<model_id>` | GET | Check cached status + file list |
| `/api/download-model/<model_id>` | GET (SSE) | Stream download progress in real-time |
| `/api/generate` | POST | `{model, voice, prompt}` → generate audio |
| `/api/audio` | GET | List all generated files with metadata |
| `/audio/<filename>` | GET | Serve audio file |
| `/` | GET | Serve `frontend/index.html` |

**SSE Download Progress Format:**
```
data: {"phase": "checking", "model": "mini"}
data: {"phase": "downloading", "file": "config.json", "progress": 100, "size": "470B", "speed": "2.81MB/s"}
data: {"phase": "downloading", "file": "kitten_tts_mini_v0_8.onnx", "progress": 78, "total_mb": 78.3, "downloaded_mb": 61.2, "speed": "87.6MB/s"}
data: {"phase": "downloading", "file": "voices.npz", "progress": 100, "size": "3.28MB", "speed": "16.3MB/s"}
data: {"phase": "ready", "message": "Model ready"}
```

> **Implementation: Capture download progress via custom `tqdm_class`.**
> `huggingface_hub.hf_hub_download()` accepts a `tqdm_class` parameter.
> Subclass `tqdm` to capture per-file progress (filename, bytes downloaded, total, speed)
> and stream it to the frontend via SSE.
>
> ```python
> from huggingface_hub import hf_hub_download
> from tqdm import tqdm
>
> class ProgressCapture(tqdm):
>     def update(self, n=1):
>         super().update(n)
>         # self.desc = filename, self.n = downloaded, self.total = total bytes
>         # Send progress via SSE to frontend here
>
> # Pre-download model files with progress tracking
> for filename in ["config.json", "model.onnx", "voices.npz"]:
>     hf_hub_download(repo_id, filename, tqdm_class=ProgressCapture)
>
> # KittenTTS loads instantly from HF cache
> m = KittenTTS(repo_id)
> ```

**File Naming Convention:**
```python
import re
from datetime import datetime

def generate_filename(prompt):
    # Clean excerpt: first 30 chars, alphanumeric only
    excerpt = re.sub(r'[^a-zA-Z0-9]+', '-', prompt[:30].lower()).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{excerpt}_{timestamp}"
# Result: "the-reports-of-my-death-are-grea_20250220_143052.wav"
```

**Metadata JSON Structure:**
```json
{
  "filename": "reports-of-my-death_20250220_143052.wav",
  "prompt": "The reports of my death are greatly exaggerated...",
  "model": "kitten-tts-mini-0.8",
  "voice": "Jasper",
  "timestamp": "2025-02-20T14:30:52",
  "inference_time": 1.234,
  "rtf": 0.15,
  "duration_seconds": 8.2,
  "sample_rate": 24000
}
```

**Port Configuration:** Auto-detect starting from 5000, increment until available. Print `🚀 Server running on http://localhost:PORT`

---

### 2. FRONTEND (`frontend/index.html`)

**Architecture:** Single file, inline CSS/JS, served by Flask

**Tech Stack:**
- Tailwind CSS v3 via CDN (custom brand palette, NO default blue/indigo)
- Vanilla JavaScript (ES6+)
- Native Web Audio API for playback
- EventSource for SSE download progress

**Design System:**
```css
:root {
  --brand-primary: #FF6B6B;      /* Coral red */
  --brand-secondary: #4ECDC4;    /* Teal */
  --brand-accent: #FFE66D;       /* Yellow */
  --brand-dark: #2C3E50;         /* Navy */
  --brand-light: #F7F9FC;        /* Off-white */
  --shadow-sm: 0 2px 4px rgba(44,62,80,0.06);
  --shadow-md: 0 4px 12px rgba(44,62,80,0.12);
  --shadow-lg: 0 8px 24px rgba(44,62,80,0.18);
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2.5rem;
}
```

**UI Layout (Mobile-First):**

```
┌─────────────────────────────────────────┐
│  🎙️  KittenTTS Studio          [≡]     │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  MODEL                          │    │
│  │  [Kitten TTS Mini    ▼]  80MB   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  VOICE                          │    │
│  │  [Jasper ●▲▼]  [Luna] [Bruno]   │    │
│  │  [Rosie] [Hugo] [Kiki] [Leo]    │    │
│  │  [Bella]                        │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  PROMPT                         │    │
│  │  ┌─────────────────────────┐    │    │
│  │  │The reports of my death..│    │    │
│  │  │are greatly exaggerated..│    │    │
│  │  └─────────────────────────┘    │    │
│  │  47 words · ~61 tokens          │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [      ✨ GENERATE AUDIO      ]        │
│                                         │
├─────────────────────────────────────────┤
│  NOW PLAYING                            │
│  ┌─────────────────────────────────┐    │
│  │  ▶️ ━━━━━━━━━━━━━━━━⚫───  0:08/0:32 │
│  │  "The reports of my death..."   │    │
│  │  Mini · Jasper · 1.2s · RTF 0.15│    │
│  │  [⬇️ Download]  [🗑️ Delete]     │    │
│  └─────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│  📚 HISTORY (12 files)                  │
│  ┌─────────────────────────────────┐    │
│  │ ▶️ │reports-of-my-death...│ 2m │    │
│  │ ▶️ │ai-taking-our-jobs... │ 1h │    │
│  │ ▶️ │this-tts-works-with...│ 3h │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

**Interactive States (All Elements):**
- **Hover:** Scale 1.02, shadow increase, color shift
- **Focus:** Ring-2 ring-brand-primary outline-none
- **Active:** Scale 0.98, darker background
- **Disabled:** Opacity 50%, cursor not-allowed
- **Loading:** Skeleton shimmer or spinner

---

### 3. DOWNLOAD PROGRESS MODAL

**Triggered:** When user clicks Generate and model not cached

```
┌─────────────────────────────────────────┐
│  📥 Downloading Model                   │
│  Kitten TTS Mini (80MB)                 │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  📄 config.json                 │    │
│  │  ████████████████░░░░  100%     │    │
│  │  ✓ 470B @ 2.81MB/s              │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  🧠 kitten_tts_mini_v0_8.onnx   │    │
│  │  ██████████████░░░░░░   78%     │    │
│  │  ↓ 61.2MB / 78.3MB @ 87.6MB/s   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  🎙️ voices.npz                  │    │
│  │  ░░░░░░░░░░░░░░░░░░░░    0%     │    │
│  │  ⏳ Waiting for model...        │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [    Cancel & Use Cached Model    ]    │
│                                         │
└─────────────────────────────────────────┘
```

**Animation Specs:**
- Progress bars: `transition: width 0.3s ease-out`
- Shimmer effect on incomplete bars: `background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)` with `animation: shimmer 1.5s infinite`
- File icons pulse when downloading
- Speed text updates every 100ms
- Auto-dismiss with success toast when complete

---

### 4. SETUP & RUNNER SCRIPTS

**`requirements.txt`:**
```
https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
flask
flask-cors
```
> **Note:** KittenTTS pulls its own dependencies (numpy, soundfile, huggingface-hub, onnxruntime, spacy, torch, etc.) — do not pin them separately to avoid version conflicts.

**`setup.bat`:**
```batch
@echo off
echo 🐱 Setting up KittenTTS Studio...

if not exist venv (
    echo Creating virtual environment...
    py -3.12 -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete! Run runner.bat to start.
pause
```

**`runner.bat`:**
```batch
@echo off
echo 🚀 Starting KittenTTS Studio...

call venv\Scripts\activate.bat

:: Find available port starting from 5000
set PORT=5000
:check_port
netstat -an | find ":%PORT%" >nul
if %ERRORLEVEL% equ 0 (
    set /a PORT+=1
    goto check_port
)

echo Found available port: %PORT%

:: Start backend in background
start "KittenTTS Backend" cmd /c "venv\Scripts\python.exe backend.py --port %PORT%"

:: Wait for server to start
timeout /t 3 /nobreak >nul

:: Open browser
start http://localhost:%PORT%

echo 🎙️ KittenTTS Studio is running at http://localhost:%PORT%
echo Press any key to stop...
pause

:: Cleanup
taskkill /FI "WINDOWTITLE eq KittenTTS Backend*" /F >nul 2>&1
```

---

### 5. MODEL CONFIGURATION

**Available Models:**

| ID | Name | Params | Size | Repository |
|----|------|--------|------|------------|
| `mini` | Kitten TTS Mini | 80M | 80MB | KittenML/kitten-tts-mini-0.8 |
| `micro` | Kitten TTS Micro | 40M | 41MB | KittenML/kitten-tts-micro-0.8 |
| `nano` | Kitten TTS Nano | 15M | 56MB | KittenML/kitten-tts-nano-0.8 |
| `nano-int8` | Kitten TTS Nano INT8 | 15M | 19MB | KittenML/kitten-tts-nano-0.8-int8 |
| `nano-fp32` | Kitten TTS Nano FP32 | 15M | ~56MB | KittenML/kitten-tts-nano-0.8-fp32 |

**Voices:** Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo

> **Note:** These voice names are aliases defined in each model's `config.json`. The underlying package voices are `expr-voice-{2-5}-{m,f}`. Voice availability may vary by model.

---

### 6. ACCEPTANCE CRITERIA

- [ ] `main.py` works exactly as original CLI tool
- [ ] `backend.py` serves API and frontend on auto-detected port
- [ ] Frontend is single `index.html` with inline Tailwind CSS (custom colors, no blue/indigo)
- [ ] Mobile-first responsive (320px to 1440px+)
- [ ] Model download shows real-time progress via SSE (3 files: config, onnx, voices)
- [ ] Generation shows loader with token count estimation
- [ ] Audio auto-plays on completion with native player
- [ ] Files saved to `/audio/` with `excerpt_timestamp` naming + JSON metadata
- [ ] History panel lists all files, clickable to replay
- [ ] `setup.bat` creates venv and installs dependencies
- [ ] `runner.bat` auto-detects port, launches backend, opens browser
- [ ] All interactive elements have hover/focus/active states
- [ ] Layered shadows and intentional spacing throughout

---

### 7. BONUS FEATURES (Optional)

- [ ] Dark mode toggle (persisted in localStorage)
- [ ] Keyboard shortcut: Ctrl+Enter to generate
- [ ] Drag-and-drop text file to populate prompt
- [ ] Audio waveform visualization during playback
- [ ] Copy prompt to clipboard from history

---
