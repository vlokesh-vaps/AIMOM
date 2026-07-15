# AI Agent Codebase Context: AIMOM

> **NOTE TO AI ASSISTANTS**: Read this document to instantly understand the `AIMOM` codebase. This serves as your system prompt and context reference when modifying, debugging, or extending the repository.

## 1. Project Purpose & Architecture
`AIMOM` is a FastAPI application that automates the generation of Minutes of Meeting (MoM) reports. 
**Data Flow**:
1. Upload Audio -> `app.py` (`POST /api/transcribe`)
2. FFmpeg conversion -> `services/audio/converter.py`
3. STT (Deepgram/NVIDIA) -> `services/stt/`
4. AI Analysis Pipeline -> `ai/pipeline/manager.py` (Text -> JSON structure)
5. Report Generation (PDF/Excel/Word) -> `reports/`

## 2. Core Constraints & Conventions
- **Pydantic Validation**: All structured LLM output is heavily validated against schemas in `ai/models/meeting.py`. Do NOT change these schemas without updating the corresponding LLM prompts (e.g. `ai/prompting/templates.py` or `six_agent_pipeline.py`) AND the downstream report generators (`reports/`).
- **Token Limits & Rate Limiting**: The app enforces strict rate limits (esp. for Groq and NVIDIA). Look at `LLM_REQUEST_THROTTLE_SECONDS` and `NVIDIA_REQUEST_THROTTLE_SECONDS` in `config/settings.py`.
- **Audio Conversion**: The system relies on `ffmpeg` binary installed on the OS. `AudioConverter` forces 16 kHz Mono PCM WAV or compresses to 48kbps MP3 if files are > 25MB.
- **Failover Logic**: When making LLM calls, use `_execute_with_transient_retry()` to gracefully handle `50x` and RateLimit errors.

## 3. The Dual AI Pipeline (Crucial!)
The backend can operate in two modes when analyzing a transcript (`POST /api/analyze`). **You must understand which pipeline you are editing.**

### A. Standard 6-Stage Pipeline (Default)
Found in `ai/pipeline/manager.py`. Uses pure Python for everything except extraction.
- **Stage 1 (Cleaner)**: `ai/stages/transcript_cleaner.py`
- **Stage 2 (Chunker)**: `ai/stages/chunking_engine.py` (splits transcript into 900-token chunks with 3 lines of overlap).
- **Stage 3 (Extractor)**: `ai/stages/chunk_extractor.py`. This is the ONLY LLM call. It uses `openai/gpt-oss-120b` (via Groq) to extract a highly detailed JSON based on the `CHUNK_EXTRACTION_USER_PROMPT`.
- **Stage 4 (Merge)**: `ai/stages/merge_engine.py`
- **Stage 5 (Validation)**: `ai/validators/validation_layer.py`

### B. Six-Agent Pipeline
Found in `ai/pipeline/six_agent_pipeline.py`. Triggered strictly when the NVIDIA provider is requested. It is an agentic workflow mapping to specific models:
- **Agent 1 (Groq - `openai/gpt-oss-120b`)**: Clean transcript.
- **Agent 2 (NVIDIA - `qwen3.5-122b`)**: Topic Segmentation.
- **Agent 3 (NVIDIA - `glm-5.2`)**: Discussion Extraction.
- **Agent 4 (NVIDIA - `glm-5.2`)**: Action Item Extraction.
- **Agent 5 (NVIDIA - `nemotron-3-ultra-550b`)**: Decision Synthesis.
- **Agent 6 (Groq - `openai/gpt-oss-120b`)**: Result Validation.

## 4. Key Pydantic Models (`ai/models/meeting.py`)
When you are prompting the LLM, the output JSON must precisely map to these models:

- **`ActionItem`**: Requires `task`, `owner`, `target_date`, `priority`, `status`, `notes`, `agenda_item`, `authority_context`, `tone_and_consequence`.
- **`DiscussionPoint`**: Requires `point`, `detailed_summary` (must be exhaustive narrative), `decision`, `task`, `assigned_to`, `deadline`, `priority`, `status`, `risks_or_concerns`, `suggestions`, `follow_up_required`, `notes`, `agenda_item`, `authority_context`, `tone_and_consequence`, `cross_topic_context`, `implicit_decision`.
- **`MeetingSummary`**: The final aggregate root returned by the pipeline and passed to `ReportManager`.

## 5. Reporting Subsystem (`reports/`)
- Located in `reports/report_manager.py`.
- It takes a `MeetingSummary` object.
- **PDF**: Uses `reports/pdf_generator.py` (Likely ReportLab or FPDF).
- **Excel**: Uses `reports/excel_generator.py` (Likely openpyxl or xlsxwriter).
- **Word**: Uses `reports/word_generator.py` (Likely python-docx).
- *Rule*: If you add a new field to `MeetingSummary` (e.g., `Sentiment`), you must manually update the PDF/Excel/Word generators to visually render that new field.

## 6. Temporary Test Scripts & NVIDIA Modularization
- The `temp/` directory previously contained standalone integration tests for NVIDIA NIM models. The exact logic from these scripts has now been integrated directly into the `AIMOM` pipeline.
- The `NvidiaAIProvider` uses a **Strategy Pattern**. It dynamically loads specific execution configurations from the `ai/providers/nvidia/models/` directory.

## 7. How to Make Changes
1. **Changing LLM Prompts**: Check `ai/prompting/templates.py` for the standard pipeline, or the `_system_prompt` static method in `ai/pipeline/six_agent_pipeline.py`.
2. **Adding an LLM Provider**: Inherit from `BaseAIProvider` in `ai/providers/base.py`, then register it in `AIManager._register_default_providers()` inside `ai/pipeline/manager.py`.
   - **Adding a new NVIDIA model**: Simply create a new `{model_name}.py` script inside `ai/providers/nvidia/models/` exporting an `execute()` method. No need to touch the core provider logic!
3. **Modifying STT behavior**: Edit `services/stt/deepgram_provider.py` or `services/stt/nvidia_provider.py`.
4. **Environment Variables**: Managed in `config/settings.py` overriding with `.env`. Add your variables there first.
