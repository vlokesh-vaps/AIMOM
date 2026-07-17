# AI Meeting Minutes & Summary Pipeline Reference

This document provides a detailed explanation of the **AI Summary and Export** pipeline. It details the translation, prompt engineering, registry dispatch, JSON repair loops, and Pydantic-based report generations.

---

## 1. Directory Structure

```
ai/
├── models/
│   ├── chunk.py        # Intermediate extraction schemas
│   └── meeting.py      # Final MeetingSummary and ActionItem schemas
├── pipeline/
│   ├── manager.py      # AIManager, registry, orchestration
│   └── six_agent_pipeline.py # FourAgentPipeline class definition
├── prompting/
│   └── templates.py    # SYSTEM_PROMPT & format_user_prompt
├── providers/
│   ├── base.py         # Abstract BaseAIProvider definition
│   ├── groq.py         # Groq LLM client
│   ├── gemini.py       # Google Gemini generateContent REST client
│   ├── nvidia/         # Modular NVIDIA strategy pattern
│   ├── ollama.py       # Local Ollama client
│   └── provider_manager.py # Centralized failover & recovery ProviderManager
├── stages/
│   ├── transcript_cleaner.py # Python text cleaner
│   └── chunking_engine.py    # Python text chunker
└── validators/
    └── validation_layer.py # Python validators and repair
```

---

## 2. Dynamic Pipeline Sequence

Below is the step-by-step description of the AI Summary execution flow:

### Step 1: Request Entry
The frontend calls `/api/analyze` with the payload:
- `title`: Meeting title string.
- `transcript`: Raw text.
- `language`: Meeting source language.
- `ai_provider`: Runtime provider selection (e.g. `nvidia`).
- `context`: Optional context supplied by the user.

### Step 2: Language Check & Translation
1. If the input language is not English, `AIManager.translate_to_english()` is invoked.
2. It sends the request to `ProviderManager` which routes to NVIDIA (primary) or Groq (fallback).
3. The LLM translates the text, maintaining speaker tags and timestamps.
4. The translated English text is written to `output/` and set as the active transcript for parsing.

### Step 3: Prompt Formatting
The user prompt is compiled using formatting templates inside `ai/prompting/templates.py`:
- Incorporates the `title`, `date`, `transcript`, and any optional `context`.
- Appends instructions enforcing structural compliance.
- Pairs it with agent-specific system instructions defined in the `FourAgentPipeline`.

### Step 4: Dispatch & LLM Request (ProviderManager Gateway)
All requests from agents route through `ProviderManager`:
1. **Primary Call**: Attempt NVIDIA first.
2. **Transient retry**: If NVIDIA fails with 429 rate limit or 5xx server error, automatically retry with exponential backoff (2s -> 4s -> 8s) up to 3 times.
3. **Failover**: If NVIDIA continues to fail, automatically switch the request to Groq with the equivalent fallback model.
4. **Health State**: If NVIDIA consecutive failures exceed the maximum count, NVIDIA is marked unhealthy, and subsequent agent requests bypass NVIDIA and directly route to Groq. 
5. **Recovery Check**: Unhealthy providers are probed at configured intervals, automatically restoring them as primary once operational again.

### Step 5: JSON Cleaning & Pydantic Validation
The raw response string is parsed using `json_object` and `json_list` methods in `FourAgentPipeline`:
1. **Regex Extraction**: Locates the first `{` and last `}` to filter preambles/explanations.
2. **Syntax Repair**: Cleans up trailing commas.
3. **Pydantic Validation**: Deserialized into a Pydantic `MeetingSummary` object by the final stages, which programmatically checks completeness, fills in empty strings with valid defaults, and auto-assigns owners.

---

## 3. Report Exporter Framework

Upon validation of `MeetingSummary`, `app.py` passes the object to `ReportManager.generate_reports()`.

### PDF Generator (`xhtml2pdf`)
- Compiles custom HTML from `reports/templates/meeting_template.html` and styles from `styles.css`.
- Swaps out brand colors with user `.env` values (`COMPANY_THEME_COLOR`).
- Renders the attendees, topics, decisions, timeline, and action tables.
- Invokes `pisa.CreatePDF()` to write the output to `reports/pdf/`.

### Excel Generator (`openpyxl`)
- Creates an Excel sheet containing formatted headers.
- Automatically maps action items to rows, setting priorities (High/Medium/Low) and formatting background colors.
- Enables filters, freezes the top header, and adjusts column widths dynamically.
- Saves the file to `reports/excel/`.

---

## 4. How to Add a New LLM Provider

To introduce a new provider (e.g., Anthropic Claude):

1. **Create Provider Class**: Add `ai/providers/anthropic.py` implementing `BaseAIProvider`.
2. **Register**: Register it inside `AIManager._register_default_providers()` and pass to `ProviderManager` constructor in `AIManager.__init__`.
3. **Configure Environment**: Add keys to `config/settings.py` and check them in `is_configured()`.
4. **Update ProviderManager model routing**: Mappings in `provider_manager.py` should be updated to select the new provider when preferred.
