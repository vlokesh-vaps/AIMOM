# AIMOM: Complete Codebase Documentation

This document provides a comprehensive architectural and overview of the `AIMOM` (AI Meeting Minutes) project. It covers the entire codebase, from audio ingestion and Speech-to-Text (STT) to the multi-agent AI pipeline and final report generation.

---

## 1. Project Overview
`AIMOM` is a FastAPI-based web service designed to generate detailed Minutes of Meeting (MoM) reports from raw audio recordings. The system handles the entire lifecycle:
1. **Audio Ingestion & Processing**: Validates, transcodes, and compresses audio files.
2. **Speech-to-Text (STT)**: Transcribes the audio into raw text using cloud or local AI providers.
3. **AI Pipeline (LLM Analysis)**: Cleans, chunks, extracts, and synthesizes the transcript into a structured format using a 4-agent workflow.
4. **Reporting**: Exports the synthesized data into formatted PDF, Excel, and Word documents.

---

## 2. Directory Structure

```text
AIMOM/
│
├── app.py                     # Main FastAPI application entry point
├── config/                    # Configuration settings (API keys, constraints, model assignments)
├── ai/                        # Core LLM processing and orchestration logic
│   ├── models/                # Pydantic data schemas (e.g., MeetingSummary, ActionItem)
│   ├── pipeline/              # LLM Pipeline Managers (AIManager, FourAgentPipeline)
│   ├── providers/             # LLM API wrappers
│   │   ├── groq.py            # Groq provider
│   │   ├── gemini.py          # Gemini provider
│   │   ├── ollama.py          # Ollama provider
│   │   ├── provider_manager.py# Centralized LLM ProviderManager with failover & recovery
│   │   └── nvidia/            # NVIDIA Provider (Modularized Strategy Pattern)
│   │       ├── provider.py    # Main NvidiaAIProvider orchestrator
│   │       └── models/        # Individual LLM configuration and execution strategies
│   ├── stages/                # Processing stages (TranscriptCleaner, ChunkingEngine)
│   ├── utils/                 # Token estimation, checkpoints, rate limiting
│   └── validators/            # Validation logic for final AI output (ValidationLayer)
├── services/                  # Audio manipulation and STT APIs
│   ├── audio/                 # FFmpeg wrappers for WAV/MP3 conversion
│   └── stt/                   # STT Providers (Deepgram, NVIDIA)
├── reports/                   # Document generation (PDF, Excel, Word)
├── utils/                     # General utilities (logging, file handling)
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
The `AIManager` (`ai/pipeline/manager.py`) orchestrates the conversion of raw transcripts into a structured `MeetingSummary` Pydantic model. It delegates execution to the **Four-Agent Pipeline** (`ai/pipeline/six_agent_pipeline.py`).

### The 4-Agent Pipeline Workflow
All LLM requests route through `ProviderManager` (`ai/providers/provider_manager.py`), ensuring robust NVIDIA-to-Groq failover, exponential backoff retries, and health monitoring.

1. **Transcript Cleaner (Python)**: Normalizes text, fixes broken lines, and removes non-verbal noise.
2. **Chunking Engine (Python)**: Splits the transcript intelligently based on a token budget (900 tokens), preserving overlapping lines so context is never lost across chunk boundaries.
3. **Agent 1 (NVIDIA Qwen/DeepSeek)**: Topic segmentation based on the meeting agenda.
4. **Agent 2 (NVIDIA GLM)**: Merged discussion and action item extraction. Uses a combined prompt to extract all items in a single LLM pass.
5. **Agent 3 (NVIDIA Nemotron)**: Synthesizes final decisions, parking lot items, and the executive summary.
6. **Agent 4 (Groq — optional, non-blocking)**: Programmatic validation of the final synthesized summary.

---

## 7. Centralized LLM Provider Management (`provider_manager.py`)
Provides a single, resilient gateway for all LLM calls across the application:
- **Primary / Fallback routing**: Routes requests to NVIDIA (primary) and automatically switches to Groq (fallback) if NVIDIA is down.
- **Retry with backoff**: Automatically retries transient errors (429 rate limits, 5xx server errors, timeouts) using exponential backoff (2s -> 4s -> 8s), immediately failing over once limits are hit without waiting for long cooldowns.
- **Health monitoring**: Continuously tracks provider health, probing unhealthy providers periodically and restoring them automatically once they recover.
- **Transient vs Permanent Error Classification**: Prevents endless retries on invalid payloads (e.g. context length exceeded) or configuration errors (e.g. bad API keys).

---

## 8. Reporting Layer (`reports/`)
The `ReportManager` receives the structured `MeetingSummary` and distributes the data to three separate generator classes:
- **`PDFGenerator`**: Creates a highly stylized, readable PDF containing the executive summary, topics, and discussions.
- **`ExcelGenerator`**: Creates a tabular Action Tracker (`.xlsx`), useful for project managers tracking tasks.
- **`WordGenerator`**: Generates a standard Microsoft Word document (`.docx`) for manual editing by the user.

---

## 9. Configuration & Utilities (`config/`, `utils/`)
- **`config/settings.py`**: Central repository for all environment variables, API keys (Groq, NVIDIA, Deepgram, Gemini), Token budgets, failover parameters, and model mappings.
- **`utils/logger.py`**: Application-wide structured logging setup.
- **`utils/file_utils.py`**: Helpers for sanitizing filenames and validating supported audio extensions.
