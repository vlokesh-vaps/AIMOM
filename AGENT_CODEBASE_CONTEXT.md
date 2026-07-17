# AI Agent Codebase Context: AIMOM

> **NOTE TO AI ASSISTANTS**: Read this document to instantly understand the `AIMOM` codebase. This serves as your system prompt and context reference when modifying, debugging, or extending the repository.

## 1. Project Purpose & Architecture
`AIMOM` is a FastAPI application that automates the generation of Minutes of Meeting (MoM) reports. 
**Data Flow**:
1. Upload Audio -> `app.py` (`POST /api/transcribe`)
2. FFmpeg conversion -> `services/audio/converter.py`
3. STT (Deepgram/NVIDIA) -> `services/stt/`
4. AI Analysis Pipeline -> `ai/pipeline/manager.py` (Text -> JSON structure via 4-agent workflow)
5. Report Generation (PDF/Excel/Word) -> `reports/`

## 2. Core Constraints & Conventions
- **Pydantic Validation**: All structured LLM output is heavily validated against schemas in `ai/models/meeting.py`. Do NOT change these schemas without updating the corresponding LLM prompts (e.g. `ai/prompting/templates.py` or `six_agent_pipeline.py`) AND the downstream report generators (`reports/`).
- **Token Limits & Rate Limiting**: The app enforces strict rate limits (esp. for Groq and NVIDIA). Look at `PROVIDER_MAX_RETRIES` and retry settings in `config/settings.py`.
- **Audio Conversion**: The system relies on `ffmpeg` binary installed on the OS. `AudioConverter` forces 16 kHz Mono PCM WAV or compresses to 48kbps MP3 if files are > 25MB.
- **Centralized Provider Layer**: Do NOT instantiate providers directly in pipeline files. All LLM calls must go through the `ProviderManager` (`ai/providers/provider_manager.py`) which manages health state, automatic failover (NVIDIA primary → Groq fallback), recovery, and retry logic.

## 3. The 4-Agent Pipeline
The backend uses a single, robust 4-agent orchestration workflow to analyze transcripts (`POST /api/analyze`):
1. **Pre-processing (Python)**:
   - **Transcript Cleaner**: `ai/stages/transcript_cleaner.py` (pure Python cleaning and formatting)
   - **Chunking Engine**: `ai/stages/chunking_engine.py` (splits transcript into 900-token chunks with 3 lines of overlap)
2. **AI Agents (LLM calls via ProviderManager)**:
   - **Agent 1 (NVIDIA Qwen/DeepSeek)**: Topic segmentation based on the meeting agenda.
   - **Agent 2 (NVIDIA GLM)**: Merged discussion + action item extraction. Uses a combined prompt to extract all items in a single LLM pass.
   - **Agent 3 (NVIDIA Nemotron)**: Synthesizes final decisions, parking lot items, and the executive summary.
   - **Agent 4 (Groq — optional, non-blocking)**: Programmatic validation of the final synthesized summary.

## 4. Key Pydantic Models (`ai/models/meeting.py`)
When prompting the LLM or validating, the output JSON maps to:
- **`ActionItem`**: Requires `task`, `owner`, `target_date`, `priority`, `status`, `notes`, `agenda_item`, `authority_context`, `tone_and_consequence`.
- **`DiscussionPoint`**: Requires `point`, `detailed_summary`, `decision`, `task`, `assigned_to`, `deadline`, `priority`, `status`, `risks_or_concerns`, `suggestions`, `follow_up_required`, `notes`, `agenda_item`, `authority_context`, `tone_and_consequence`, `cross_topic_context`, `implicit_decision`.
- **`MeetingSummary`**: The final aggregate root passed to `ReportManager`.

## 5. Reporting Subsystem (`reports/`)
- Located in `reports/report_manager.py`.
- It takes a `MeetingSummary` object.
- **PDF**: Uses `reports/pdf_generator.py`
- **Excel**: Uses `reports/excel_generator.py`
- **Word**: Uses `reports/word_generator.py`

## 6. NVIDIA Modularization & Strategy Pattern
- The `NvidiaAIProvider` uses a **Strategy Pattern**. It dynamically loads specific execution configurations from the `ai/providers/nvidia/models/` directory (e.g. `deepseek.py`, `glm.py`, `nemotron.py`).
- Adding a new NVIDIA model is as simple as adding a strategy Python file inside `ai/providers/nvidia/models/` implementing the `execute` function.

## 7. How to Make Changes
1. **Changing LLM Prompts**: Check `ai/prompting/templates.py` or the `_system_prompt` static method in `ai/pipeline/six_agent_pipeline.py`.
2. **Adding an LLM Provider**: Inherit from `BaseAIProvider` in `ai/providers/base.py`, register it in `AIManager._register_default_providers()`, and update `ProviderManager` to support it.
3. **Modifying STT behavior**: Edit `services/stt/deepgram_provider.py` or `services/stt/nvidia_provider.py`.
4. **Environment Variables**: Centralized in `config/settings.py` and `.env`.
