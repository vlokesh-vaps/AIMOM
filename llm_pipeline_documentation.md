# AIMOM: LLM Pipeline Architecture Documentation

Based on a detailed analysis of the `AIMOM` codebase, here is a comprehensive breakdown of how Large Language Models (LLMs) are utilized within the meeting analysis pipeline.

## Overview

The application is a FastAPI-based backend that handles transcription (STT) and subsequent AI analysis of meeting transcripts to generate Minutes of Meeting (MoM) reports. The core of the LLM orchestration is managed by the `AIManager` located in `ai/pipeline/manager.py`.

The AI analysis phase executes in one of two distinct operational modes depending on the selected provider:
1. **Six-Agent Pipeline (`six_agent_pipeline.py`)**: A multi-agent workflow utilizing a mix of NVIDIA and Groq LLMs for fine-grained task separation.
2. **Standard 6-Stage Pipeline (`manager.py`)**: A hybrid approach where most stages are handled deterministically in pure Python, and the LLM is only utilized for the heavy-lifting extraction phase.

---

## 1. The Six-Agent Pipeline (`SixAgentPipeline`)
When the NVIDIA provider is selected and Groq is available, the system triggers `SixAgentPipeline`. This pipeline treats the extraction process as a sequential, multi-agent workflow where each agent is an LLM with a specific, focused prompt. 

The pipeline chunks the transcript first (to stay within context windows) and passes each chunk through these LLM agents:

- **Agent 1 (Transcript Cleanup):** 
  - **Provider:** Groq
  - **Role:** Cleans raw text, preserves speaker names/facts, and normalizes the format.
- **Agent 2 (Topic Segmentation):** 
  - **Provider:** NVIDIA
  - **Role:** Segments the transcript into topics based on the agenda.
- **Agent 3 (Discussion Extraction):** 
  - **Provider:** NVIDIA
  - **Role:** Extracts discussion points mapped to the agenda, providing factual summaries.
- **Agent 4 (Action Extraction):** 
  - **Provider:** NVIDIA
  - **Role:** Extracts action items, strictly mapping task owners and target dates to the transcript without hallucinating.
- **Agent 5 (Decision Synthesis):** 
  - **Provider:** NVIDIA
  - **Role:** Takes the outputs of Agents 2, 3, and 4 to synthesize final decisions and the executive summary.
- **Agent 6 (Validation):** 
  - **Provider:** Groq
  - **Role:** Validates the final synthesized summary for missing facts or logical inconsistencies.

---

## 2. The Standard 6-Stage Pipeline
If a provider other than NVIDIA is selected (e.g., Groq default, or Gemini), the system uses the standard pipeline defined directly in `AIManager`. This approach minimizes LLM API calls by relying on deterministic Python logic for data structuring.

- **Stage 1 (Transcript Cleaner - Pure Python):** Normalizes text and reduces noise without using an LLM.
- **Stage 2 (Chunking Engine - Pure Python):** Intelligently splits the transcript into token-aware chunks preserving overlap context.
- **Stage 3 (Chunk Extractor - LLM):** 
  - **Provider:** Groq (default) or other configured LLMs.
  - **Role:** The **only LLM step** in this pipeline. It processes chunks to extract structured JSON data containing discussion points, actions, and decisions.
- **Stage 4 (Merge Engine - Pure Python):** Combines the extracted JSON chunks from Stage 3 into a cohesive summary, resolving duplicates.
- **Stage 5 (Validation Layer - Pure Python):** Verifies the completeness of the merged summary programmatically.
- **Stage 6:** Final result delivery.

---

## 3. Resilience and Failover
The `AIManager` implements a robust queueing and failover mechanism:
- **Provider Priority:** Groq -> NVIDIA -> Gemini.
- **Transient Retry:** The system gracefully handles `RateLimit`, `Timeout`, and `50x` errors using an exponential backoff (`_execute_with_transient_retry`). 
- **Context Window Adaptation:** If an LLM returns a truncated response, the pipeline automatically detects it and retries with an expanded `max_tokens` budget.

## 4. Role of the Temp Test Scripts and NVIDIA Integration
The files located in the `temp/` directory (`glm-5.2.py`, `nemotron-3.py`, `qwen.py`) were originally standalone diagnostic scripts. 
- **Recent Update:** The exact working methods and API configurations (e.g., using `openai` client, `stream=True`, and `extra_body` kwargs) from these test scripts have been seamlessly integrated into the core pipeline.
- The `NvidiaAIProvider` is now completely modularized into the `ai/providers/nvidia/` directory. It uses a **Strategy Pattern**, dynamically loading specific configurations from `ai/providers/nvidia/models/`. This architecture makes adding new NVIDIA models as simple as dropping a new Python file into the models folder.
