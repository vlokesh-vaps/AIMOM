# AIMOM: Complete Codebase Documentation

This document provides a comprehensive architectural and operational overview of the `AIMOM` (AI Meeting Minutes) project. It covers the entire codebase, from audio ingestion and Speech-to-Text (STT) to the multi-agent AI pipeline and final report generation.

---

## 1. Project Overview
`AIMOM` is a FastAPI-based web service designed to generate detailed Minutes of Meeting (MoM) reports from raw audio recordings. The system handles the entire lifecycle:
1. **Audio Ingestion & Processing**: Validates, transcodes, and compresses audio files.
2. **Speech-to-Text (STT)**: Transcribes the audio into raw text using cloud or local AI providers.
3. **AI Pipeline (LLM Analysis)**: Cleans, chunks, extracts, and synthesizes the transcript into a structured format (Discussion Points, Action Items, Decisions).
4. **Reporting**: Exports the synthesized data into formatted PDF, Excel, and Word documents.

---

## 2. Directory Structure

```text
AIMOM/
│
├── app.py                     # Main FastAPI application entry point
├── config/                    # Configuration settings (API keys, constraints, model map)
├── ai/                        # Core LLM processing and orchestration logic
│   ├── models/                # Pydantic data schemas (e.g., MeetingSummary, ActionItem)
│   ├── pipeline/              # LLM Pipeline Managers (Standard vs. Six-Agent)
│   ├── providers/             # LLM API wrappers
│   │   ├── groq.py            # Groq provider
│   │   ├── gemini.py          # Gemini provider
│   │   └── nvidia/            # NVIDIA Provider (Modularized Strategy Pattern)
│   │       ├── provider.py    # Main NvidiaAIProvider orchestrator
│   │       └── models/        # Individual LLM configuration and execution scripts
│   ├── stages/                # Individual processing stages (Chunking, Extracting, Merging)
│   ├── utils/                 # Token estimation, checkpoints, rate limiting
│   └── validators/            # Validation logic for final AI output
├── services/                  # Audio manipulation and STT APIs
│   ├── audio/                 # FFmpeg wrappers for WAV/MP3 conversion
│   └── stt/                   # STT Providers (Deepgram, NVIDIA)
├── reports/                   # Document generation (PDF, Excel, Word)
├── utils/                     # General utilities (logging, file handling)
├── temp/                      # Temporary files and standalone LLM test scripts
├── output/                    # Saved raw transcripts
└── templates/                 # HTML templates for the frontend UI
```

---

## 3. The API & Web Layer (`app.py`)
`app.py` serves as the central router and controller.
- **`GET /`**: Serves the frontend UI (`templates/index.html`).
- **`POST /api/transcribe`**: Accepts an audio file upload, processes the audio (FFmpeg), and calls the appropriate STT provider to return a text transcript.
- **`POST /api/analyze`**: Accepts a text transcript and contextual parameters (title, attendees, agenda), routes it through the AI Pipeline, and triggers the `ReportManager`. Returns URLs to download the generated files.
- **Download Endpoints**: Serves the generated PDF, Excel, and Word files from the `/reports/` directory.

---

## 4. Audio Processing (`services/audio/`)
The `AudioConverter` uses system `FFmpeg` to normalize audio for STT APIs.
- **Transcoding**: Ensures the audio is exactly 16 kHz Mono PCM WAV (`convert_to_wav`), bypassing transcoding if the file is already formatted correctly (checked via `ffprobe`).
- **Compression**: For large files (>25MB), it compresses the audio to a 48kbps MP3 (`compress_to_mp3`) to prevent cloud API timeouts.

---

## 5. Speech-to-Text (STT) Layer (`services/stt/`)
Orchestrated by the `ProviderManager`, the system supports multiple STT engines:
- **Deepgram (`DeepgramProvider`)**: Used for fast cloud-based transcription.
- **NVIDIA (`NvidiaProvider`)**: Used for highly accurate NIM-based transcriptions.
Each provider inherits from `BaseSTTProvider`, ensuring a standardized `.transcribe()` interface.

---

## 6. AI Analysis Pipeline (`ai/`)
This is the intellectual core of the application. The `AIManager` (`ai/pipeline/manager.py`) orchestrates the conversion of raw transcripts into a structured `MeetingSummary` Pydantic model. Depending on the selected LLM provider, it uses one of two architectures:

### A. The Standard 6-Stage Pipeline
Used as the default (e.g., when Groq or Gemini is selected). This minimizes LLM usage to a single heavy extraction step:
1. **Transcript Cleaner (Python)**: Normalizes text, fixes broken lines, and removes non-verbal noise.
2. **Chunking Engine (Python)**: Splits the transcript intelligently based on a token budget, preserving overlapping lines so context is never lost across chunk boundaries.
3. **Chunk Extractor (LLM)**: Passes each chunk to an LLM (e.g., Groq `llama-3.3-70b-versatile` or `openai/gpt-oss-20b`) to extract discussion points, action items, and decisions in a strict JSON format.
4. **Merge Engine (Python)**: Combines the extracted JSON objects, resolving duplicate action items or split topics.
5. **Validation Layer (Python)**: Verifies the completeness of the merged summary programmatically.
6. **Final Result**: Yields the validated summary.

### B. The Six-Agent Pipeline (`ai/pipeline/six_agent_pipeline.py`)
Triggered when the **NVIDIA** provider is selected (requires Groq as a fallback for specific tasks). This treats the extraction process as a multi-agent workflow:
- **Agent 1 (Groq)**: Transcript cleanup and entity preservation.
- **Agent 2 (NVIDIA)**: Topic segmentation based on the meeting agenda.
- **Agent 3 (NVIDIA)**: Discussion extraction mapped strictly to topics.
- **Agent 4 (NVIDIA)**: Action extraction with strict constraints (no hallucinated owners or dates).
- **Agent 5 (NVIDIA)**: Decision synthesis and executive summary generation.
- **Agent 6 (Groq)**: Final validation for missing facts or logical inconsistencies.

*Note on Resilience: The `AIManager` has a robust exponential backoff system handling rate limits, timeouts, and dynamically increasing the token budget if an LLM returns a truncated response.*

---

## 7. Reporting Layer (`reports/`)
The `ReportManager` receives the structured `MeetingSummary` and distributes the data to three separate generator classes:
- **`PDFGenerator`**: Creates a highly stylized, readable PDF containing the executive summary, topics, and discussions.
- **`ExcelGenerator`**: Creates a tabular Action Tracker (`.xlsx`), useful for project managers tracking tasks.
- **`WordGenerator`**: Generates a standard Microsoft Word document (`.docx`) for manual editing by the user.

---

## 8. Configuration & Utilities (`config/`, `utils/`)
- **`config/settings.py`**: Central repository for all environment variables, API keys (Groq, NVIDIA, Deepgram, Gemini), Token budgets, and model mappings.
- **`utils/logger.py`**: Application-wide structured logging setup.
- **`utils/file_utils.py`**: Helpers for sanitizing filenames and validating supported audio extensions.

---

## 9. Diagnostic Scripts (`temp/`)
Files located in the `temp/` folder (e.g., `glm-5.2.py`, `nemotron-3.py`, `qwen.py`) originated as standalone scripts used during development. **Update**: The exact working configurations and OpenAI client executions from these test scripts have now been formally integrated into the pipeline via the modular `ai/providers/nvidia/models/` strategy pattern architecture.
