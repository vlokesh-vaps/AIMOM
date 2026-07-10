# Production AI Pipeline Architecture

The AI Meeting Minutes pipeline is organized by responsibility so each stage can
be tested, replaced, and monitored independently.

## Runtime Flow

Transcript -> Transcript Cleaner -> Intelligent Chunking Engine -> Groq Chunk
Extractor (`openai/gpt-oss-20b`) -> Python Merge Engine -> Pydantic Validation
-> Ollama Final Reviewer (`gemma4:latest`) -> PDF / Excel / Database.

## Package Layout

- `ai/pipeline/`: orchestration only. `AIManager` wires stages together.
- `ai/stages/`: single-purpose pipeline stages.
- `ai/providers/`: provider clients and provider error classes.
- `ai/models/`: Pydantic models for chunk extraction and final reports.
- `ai/validators/`: schema validation and repair.
- `ai/prompting/`: extraction and review prompts.
- `ai/utils/`: checkpoints, rate limiting, token estimation, and parsing helpers.

The old flat modules, such as `ai.manager` and `ai.schemas`, remain as
compatibility wrappers while application code migrates to the production package
paths.

## Reliability Rules

- Stage 3 extraction uses Groq only and emits compact factual JSON.
- Chunk size defaults to 900 tokens and is clamped to the 700-900 target range.
- Truncated or malformed chunk output triggers recursive chunk splitting.
- Completed chunks are checkpointed to disk under `temp/ai_checkpoints`.
- Provider pacing is controlled by a request scheduler tracking RPM, TPM, and
  concurrency.
- Merging is deterministic Python logic and does not call an LLM.
- Ollama final review is allowed to polish wording only; output is rejected if
  it removes required facts.
