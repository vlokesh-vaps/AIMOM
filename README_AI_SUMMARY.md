# AI Meeting Minutes & Summary Pipeline Reference

This document provides a detailed explanation of the **AI Summary and Export** pipeline. It details the translation, prompt engineering, registry dispatch, JSON repair loops, and Pydantic-based report generations.

---

## 1. Directory Structure

```
ai/
├── provider.py         # Abstract BaseAIProvider definition
├── manager.py          # AIManager, registry, failover engine
├── prompts.py          # SYSTEM_PROMPT & user prompts definitions
├── parser.py           # JSON stripping, regex cleaning, schema validation
├── schemas.py          # Pydantic MeetingSummary and ActionItem schemas
├── groq_provider.py    # Groq LLaMA client
├── gemini_provider.py  # Google Gemini generateContent REST client
├── nvidia_provider.py  # NVIDIA NIM completions client
└── ollama_provider.py  # Local Ollama client
```

---

## 2. Dynamic Pipeline Sequence

Below is the step-by-step description of the AI Summary execution flow:

### Step 1: Request Entry
The frontend calls `/api/analyze` with the payload:
- `title`: Meeting title string.
- `transcript`: Raw text.
- `language`: Meeting source language.
- `ai_provider`: Runtime provider selection (`groq`, `gemini`, `nvidia`, `ollama`).
- `context`: Optional context supplied by the user.

### Step 2: Language Check & Translation
1. If the input language is not English, `AIManager.translate_to_english()` is invoked.
2. It constructs a system translation instruction and selects the primary provider (with fallbacks).
3. The LLM translates the text, maintaining speaker tags and timestamps.
4. The translated English text is written to `output/` and set as the active transcript for parsing.

### Step 3: Prompt Formatting
The user prompt is compiled in `ai/prompts.py`:
- Incorporates the `title`, `date`, `transcript`, and any optional `context`.
- Appends instructions enforcing structural compliance.
- Pairs it with `SYSTEM_PROMPT` containing the Pydantic schema keys and descriptions.

### Step 4: Dispatch & LLM Request
`AIManager.analyze_meeting()` builds a queue of providers:
1. Enqueues the selected provider as primary.
2. Appends other configured providers (those with valid API keys in `.env`) as fallback targets.
3. Iterates over the queue:
   - Sets `responseMimeType: "application/json"` (Gemini) or `"format": "json"` (Ollama) to force structured outputs when supported.
   - Executes the HTTP post request.

### Step 5: JSON Cleaning & Regex Repair
The raw response string is processed in `ai/parser.py`:
1. **Regex Extraction**: Filters preambles or notes by locating the first `{` and last `}` using `re.search(r"(\{.*\})", raw_response, re.DOTALL)`.
2. **Syntax Repair**: Cleans up trailing commas, missing quotes around keys, and minor structural syntax glitches.
3. **Pydantic Validation**: Deserializes the string into a Pydantic `MeetingSummary` object:
   ```python
   summary = MeetingSummary.model_validate_json(cleaned_json)
   ```
4. **Retry Loop**: If validation fails (missing required fields, format mismatches), it sleeps (exponential backoff) and retries the request up to 3 times before failing over to the next provider.

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

1. **Create Provider Class**: Add `ai/anthropic_provider.py` implementing `BaseAIProvider`:
   ```python
   from ai.provider import BaseAIProvider

   class AnthropicAIProvider(BaseAIProvider):
       def generate_text(self, system_prompt, user_prompt) -> str:
           # Request logic for Anthropic Messages API
           return raw_text
       
       def generate_summary(self, title, date, transcript, speaker_transcript=None) -> str:
           # Format user prompt and return self.generate_text()
   ```

2. **Register**: Register it inside `AIManager._register_default_providers()` in [manager.py](file:///c:/Users/Vaps/PycharmProjects/AIMOM/ai/manager.py):
   ```python
   self.register("anthropic", AnthropicAIProvider())
   ```

3. **Configure Environment**: Add keys (`ANTHROPIC_API_KEY`) to `config/settings.py` and check them in `is_configured()`.
