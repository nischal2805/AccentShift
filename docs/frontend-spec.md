# DP6: Accent Shift — Frontend Specification (Next.js)

## Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui components
- WaveSurfer.js (waveform visualisation + playback)
- Axios (API calls)
- React Hook Form + Zod (upload form validation)

## Pages

### `/` — Main Conversion Page

Single-page app. No routing needed beyond this.

**Layout:**

```
┌─────────────────────────────────────────────────────┐
│  Header: "Accent Shift" logo + tagline              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   UPLOAD PANEL      │  │   OUTPUT PANEL      │  │
│  │                     │  │                     │  │
│  │  Drag & drop zone   │  │  (empty until job   │  │
│  │  or click to browse │  │   completes)        │  │
│  │                     │  │                     │  │
│  │  Waveform preview   │  │  Waveform player    │  │
│  │  (source audio)     │  │  (converted audio)  │  │
│  │                     │  │                     │  │
│  │  Accent selector    │  │  Emotion metrics    │  │
│  │  (dropdown)         │  │  bar chart          │  │
│  │                     │  │                     │  │
│  │  [Convert] button   │  │  [Download] button  │  │
│  └─────────────────────┘  └─────────────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Metrics row (shown after conversion)       │   │
│  │  Emotion Sim: 0.91  WER: 4.2%  MOS: 4.1    │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Components

### `AudioUploader`
- Drag-and-drop zone using native HTML5 drag events
- Accepts: `.wav`, `.mp3`, `.flac`, `.m4a` (convert to WAV server-side)
- Max file size: 50MB
- On file select: render WaveSurfer waveform immediately (client-side, before upload)
- Show duration, sample rate, file name below waveform

### `AccentSelector`
- shadcn/ui `Select` component
- Options: Indian English, Chinese English, Japanese English, British English, American English
- Value maps to accent key sent to API: `indian_english`, `chinese_english`, etc.

### `ConvertButton`
- Disabled until file selected + accent selected
- On click: POST to `/api/convert` with FormData
- Shows spinner + "Converting..." during processing
- Processing time estimate shown: "~15-30 seconds"

### `WaveformPlayer`
- WaveSurfer.js instance
- Play/pause button, seek bar, duration display
- Two instances: one for source (upload panel), one for output (output panel)
- Output waveform renders only after conversion completes

### `EmotionMetrics`
- Three horizontal bars: Valence, Arousal, Dominance
- Two values per bar: source (grey) vs output (brand colour) — overlaid
- Shows how well emotion was preserved visually
- Tooltip: "Higher similarity = better emotion preservation"

### `MetricsRow`
- Three stat cards: Emotion Similarity (%), WER (%), MOS Score (/5.0)
- Colour coded: green if good (sim>0.85, wer<10%, mos>3.8), yellow if marginal, red if poor

## API Routes (Next.js `/app/api/`)

### `POST /api/convert`
Proxies to FastAPI backend. Reason: avoid CORS issues, keep backend URL server-side only.

```typescript
// app/api/convert/route.ts
// Receives: FormData { audio: File, target_accent: string }
// Forwards to: process.env.BACKEND_URL/convert
// Returns: { audio_url: string, metrics: ConversionMetrics }
```

### `GET /api/accents`
Returns available accent list from backend.

## Environment Variables

```
NEXT_PUBLIC_APP_NAME=Accent Shift
BACKEND_URL=http://localhost:8000   # internal, never exposed to client
```

## Key Behaviours

- Conversion is synchronous (FastAPI waits, returns audio). No polling needed unless audio > 2 mins.
- Converted audio returned as base64 or binary blob, rendered directly in WaveSurfer (no file URL needed)
- If conversion fails, show toast error with the error message from backend
- Mobile responsive: stack panels vertically on <768px

## Design Tokens

- Font: Inter
- Primary colour: #6366F1 (indigo-500)
- Background: #0F0F0F (near black)
- Card background: #1A1A1A
- Text: #F5F5F5
- Success: #22C55E, Warning: #EAB308, Error: #EF4444
- Dark theme only — no light mode toggle needed
