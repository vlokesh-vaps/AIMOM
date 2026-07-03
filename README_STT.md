# Speech-to-Text (STT) Transcription Pipeline Reference

This document provides a detailed, technical explanation of the **Speech-to-Text (STT)** pipeline, from the audio input to the persistent text output. Use this guide to debug transcription issues or add new STT engines.

---

## 1. Directory Structure

```
services/
├── audio/
│   ├── recorder.py         # Handles mic voice capturing via sounddevice
│   └── converter.py        # Transcodes files to 16kHz mono WAV via FFmpeg
└── stt/
    ├── base.py             # Abstract BaseSTTProvider interface
    ├── nvidia_provider.py  # NVIDIA Riva gRPC client (Parakeet/Whisper)
    ├── deepgram_provider.py # Deepgram SDK client (Nova-3/Nova-2)
    └── provider_manager.py # Models registry / router
```

---

## 2. Dynamic Pipeline Sequence

Below is the step-by-step description of the transcription flow:

### Step 1: Request Initialization
When a user uploads a file or stops recording, the frontend issues a `POST` request to `/api/transcribe` with the parameters:
- `file`: Audio binary.
- `engine`: The selected model name (e.g., `NVIDIA Whisper Large v3` or `Deepgram Nova-3`).
- `language`: Target language string (e.g., `Kannada`, `Hindi`, `English`).
- `title`: Meeting title used for file names.

### Step 2: Engine Lookup
`app.py` queries the global `ProviderManager` instance using:
```python
stt_provider = provider_manager.get_provider(engine)
```
The manager parses the engine string and retrieves the registered `BaseSTTProvider` subclass (either `NvidiaProvider` or `DeepgramProvider`).

### Step 3: WAV Conversion / Transcoding
Some engines require a strict input format (e.g., 16kHz, mono, 16-bit PCM WAV).
1. `app.py` checks `stt_provider.requires_conversion_to_wav(file_path, engine)`.
2. If `True` (typically for NVIDIA Riva Parakeet/CTC), the path is routed to `AudioConverter.convert_to_wav()`.
3. `AudioConverter` runs an FFmpeg process in a subprocess:
   ```bash
   ffmpeg -y -i <input_path> -acodec pcm_s16le -ac 1 -ar 16000 <output_path>
   ```
4. The output path updates to the temporary `.wav` file.

### Step 4: Provider Dispatch
`stt_provider.transcribe(audio_path, language, engine)` is executed:
- **NVIDIA Riva (gRPC)**:
  - Establishes a gRPC channel with the Riva server at `grpc.nvcf.nvidia.com:443`.
  - Configures the request with the specific `function_id`, model name, language, and sample rate.
  - Sends the raw PCM bytes chunk-by-chunk over a streaming request or as a single batch request.
  - Receives back transcriptions and merges text fragments.
- **Deepgram (REST API/SDK)**:
  - Invokes the Deepgram Python SDK.
  - Sends the file stream to the completions endpoint using the `nova-3` or `nova-2` model parameter.
  - Sets options like `smart_format=True` and `diarize=True` (speaker identification).

### Step 5: Persistence & Cleanup
1. The return value is wrapped in a `TranscriptionResult` object containing the transcript text and duration.
2. The raw text is saved to `output/{sanitized_title}_{timestamp}.txt`.
3. All temporary files in `temp/` created during the upload and conversion stages are deleted.
4. The transcription text is sent back to the frontend in a JSON response.

---

## 3. How to Add a New STT Provider

To introduce a new transcription engine (e.g., OpenAI Whisper API):

1. **Create Provider File**: Write `services/stt/openai_provider.py` implementing the `BaseSTTProvider` interface:
   ```python
   from services.stt.base import BaseSTTProvider, TranscriptionResult

   class OpenAIWhisperProvider(BaseSTTProvider):
       def transcribe(self, file_path, language, model_name) -> TranscriptionResult:
           # Implement API request/processing logic
           return TranscriptionResult(text=text, duration_seconds=duration)

       def requires_conversion_to_wav(self, file_path, model_name) -> bool:
           return False # Whisper accepts standard MP3/M4A directly
   ```

2. **Register Engine**: In [app.py](file:///c:/Users/Vaps/PycharmProjects/AIMOM/app.py), import the provider, instantiate it, and register it under its model options:
   ```python
   openai_provider = OpenAIWhisperProvider()
   provider_manager.register("OpenAI Whisper-1", openai_provider)
   ```

3. **Update Settings**: Add the display name to `config/settings.py` or `.env` list so it displays in the frontend select options.

---

## 4. Error Handling Strategy

- **ModuleNotFoundError / Dependencies**: The audio recorders use `sounddevice` which depends on `portaudio`. If you get loading errors, ensure the appropriate system libraries are installed.
- **FFmpeg Errors**: If audio conversion fails with a `FileNotFoundError`, FFmpeg is not installed on the system PATH. Ensure it is accessible globally.
- **gRPC Connection Timeouts**: NVIDIA Riva is hosted on NVIDIA Cloud Functions. The gRPC server times out after `AI_TIMEOUT` (default: 60s). Ensure network traffic allows outbound connections on port `443`.
