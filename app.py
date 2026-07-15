"""AI Meeting Minutes — FastAPI web service with Phase 3 (AI analysis) and Phase 4 (Reporting) integration.

Serves REST APIs for transcribing audio files, executing LLM analysis,
generating reports (PDF + Excel), and serving the web frontend.
"""

import os
import shutil
import time
from typing import Optional
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from config.settings import (
    DEEPGRAM_MODEL_MAP,
    NVIDIA_MODEL_MAP,
    LANGUAGE_OPTIONS,
    TEMP_DIR,
    OUTPUT_DIR,
)
from services.stt.deepgram_provider import DeepgramProvider
from services.stt.nvidia_provider import NvidiaProvider
from services.stt.provider_manager import ProviderManager
from ai.pipeline.manager import AIManager
from reports.report_manager import ReportManager
from utils.file_utils import ensure_directories, generate_filename, is_supported_audio
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize directories
ensure_directories()

# Initialize speech-to-text providers
provider_manager = ProviderManager()
nvidia_provider = NvidiaProvider()
for display_name in NVIDIA_MODEL_MAP:
    provider_manager.register(display_name, nvidia_provider)

deepgram_provider = DeepgramProvider()
for display_name in DEEPGRAM_MODEL_MAP:
    provider_manager.register(display_name, deepgram_provider)

# Initialize AI and reporting services
ai_manager = AIManager()
report_manager = ReportManager()

app = FastAPI(title="AI Meeting Minutes API")

# Ensure templates directory exists for index.html
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)


class AnalyzeRequest(BaseModel):
    """Request payload schema for executing transcript analysis."""
    title: str
    transcript: str
    language: str = "English"
    ai_provider: Optional[str] = None
    context: Optional[str] = None
    attendees: Optional[str] = None  # Comma-separated attendee names entered by user
    agenda: Optional[str] = None  # Meeting agenda pasted by user after STT
    meeting_date: Optional[str] = None
    absents: Optional[str] = None
    chaired_by: Optional[str] = None
    organization: Optional[str] = None



@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the web application frontend UI."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/api/engines")
async def get_engines():
    """Return all available speech-to-text engines/models."""
    return {"engines": provider_manager.get_available_providers()}


@app.get("/api/languages")
async def get_languages():
    """Return list of supported languages."""
    return {"languages": LANGUAGE_OPTIONS}






@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: str = Form(...),
    language: str = Form("English"),
    title: str = Form("meeting"),
):
    """Upload, convert, and transcribe meeting audio file immediately.

    Deletes the temporary file immediately after parsing and saves text output.
    """
    logger.info("Received transcription request: engine=%s, lang=%s, file=%s", engine, language, file.filename)

    # 1. Validate file extension
    file_path = Path(file.filename)
    if not is_supported_audio(file_path):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {file_path.suffix}",
        )

    # 2. Get STT provider
    try:
        stt_provider = provider_manager.get_provider(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. Save uploaded file to temp directory
    temp_filename = generate_filename(title, file_path.suffix)
    temp_file_path = TEMP_DIR / temp_filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved upload temporarily to %s", temp_file_path)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # 4. Transcribe and clean up the audio file
    transcription_result = None
    transcription_error = None
    try:
        # Convert to WAV if needed
        if stt_provider.requires_conversion_to_wav(temp_file_path, engine):
            logger.info("Audio conversion to WAV required for engine %s", engine)
            from services.audio.converter import AudioConverter
            final_audio_path = AudioConverter.convert_to_wav(temp_file_path, title=title)
            # If a new converted file was created, we track it for deletion
            converted_path = final_audio_path if final_audio_path != temp_file_path else None
        else:
            # Check size to determine if we should compress (for cloud providers like Deepgram)
            file_size_mb = temp_file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25.0:
                logger.info("File size is %.2f MB (> 25 MB). Compressing to mono MP3 to avoid cloud timeout...", file_size_mb)
                from services.audio.converter import AudioConverter
                final_audio_path = AudioConverter.compress_to_mp3(temp_file_path, title=title)
                converted_path = final_audio_path
            else:
                logger.info("No audio conversion or compression needed for %s (%.2f MB)", temp_file_path.name, file_size_mb)
                final_audio_path = temp_file_path
                converted_path = None

        # Run STT
        logger.info("Running transcription...")
        transcription_result = stt_provider.transcribe(final_audio_path, language, engine)

        # Clean up converted/compressed file if one was created
        if converted_path and converted_path.exists():
            try:
                os.remove(converted_path)
                logger.info("Deleted converted/compressed temp file: %s", converted_path)
            except Exception as e:
                logger.warning("Could not delete converted/compressed temp file: %s", e)

    except Exception as e:
        logger.exception("Error during transcription process")
        transcription_error = str(e)
    finally:
        # Guarantee deletion of the uploaded temporary file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.info("Deleted temporary upload file: %s", temp_file_path)
            except Exception as e:
                logger.warning("Could not delete temporary upload file: %s", e)

    if transcription_error:
        raise HTTPException(status_code=500, detail=transcription_error)

    # 5. Save the transcript content to output directory
    transcript_filename = generate_filename(title, ".txt")
    transcript_path = OUTPUT_DIR / transcript_filename
    try:
        transcript_path.write_text(transcription_result.text, encoding="utf-8")
        logger.info("Saved transcript to: %s", transcript_path)
    except Exception as e:
        logger.warning("Failed to save transcript to disk: %s", e)

    return {
        "text": transcription_result.text,
        "duration_seconds": transcription_result.duration_seconds,
        "engine": engine,
        "language": language,
        "saved_path": str(transcript_path),
    }


@app.post("/api/analyze")
async def analyze_meeting_transcript(req: AnalyzeRequest):
    """Execute Phase 3: AI analysis and Phase 4: Report generation on transcript text."""
    logger.info("Received analysis request for title: '%s', language: '%s'", req.title, req.language)
    
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty.")

    # Translate first if not English
    translated_transcript = req.transcript
    is_translated = False
    if req.language.lower() not in ("english", "auto"):
        try:
            translated_transcript = ai_manager.translate_to_english(
                req.transcript, req.language, provider_override=req.ai_provider
            )
            if translated_transcript != req.transcript:
                is_translated = True
                # Save translated transcript to output folder
                trans_filename = generate_filename(req.title, "_translated_en.txt")
                trans_path = OUTPUT_DIR / trans_filename
                trans_path.write_text(translated_transcript, encoding="utf-8")
                logger.info("Saved translated English transcript to: %s", trans_path)
        except Exception as e:
            logger.warning("Translation to English failed, falling back to original transcript: %s", e)

    # If user provided additional context, prepend it to the transcript
    analysis_transcript = translated_transcript
    if req.context and req.context.strip():
        analysis_transcript = f"[Additional Context: {req.context.strip()}]\n\n{translated_transcript}"
        logger.info("Appended user context to transcript for AI analysis")

    meeting_date = (req.meeting_date or "").strip() or datetime.now().strftime("%Y-%m-%d")
    
    try:
        summary_result = ai_manager.analyze_meeting(
            title=req.title,
            date=meeting_date,
            transcript=analysis_transcript,
            provider_override=req.ai_provider,
            attendees=req.attendees or None,
            agenda=req.agenda or None,
        )
        summary_result.meeting_date = meeting_date
        summary_result.absents = (req.absents or "").strip() or "Nil"
        summary_result.chaired_by = (req.chaired_by or "").strip() or ""
        summary_result.organization = (req.organization or "").strip() or ""
    except Exception as e:
        logger.exception("AI meeting analysis failed")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")

    try:
        report_paths = report_manager.generate_reports(summary_result)
        pdf_path = Path(report_paths["pdf"])
        excel_path = Path(report_paths["excel"])
        word_path = Path(report_paths["word"]) if report_paths.get("word") else None
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return {
        "summary": summary_result.model_dump(),
        "pdf_url": f"/api/reports/pdf/{pdf_path.name}",
        "excel_url": f"/api/reports/excel/{excel_path.name}",
        "word_url": f"/api/reports/word/{word_path.name}" if word_path else None,
        "translated_text": translated_transcript if is_translated else None
    }


@app.get("/api/transcripts")
async def list_saved_transcripts():
    """List all saved transcripts (.txt files) in the output directory."""
    if not OUTPUT_DIR.exists():
        return {"transcripts": []}

    transcripts = []
    for file in OUTPUT_DIR.glob("*.txt"):
        # Skip translation files to keep list clean
        if file.name.endswith("_translated_en.txt"):
            continue
        
        stat = file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        
        # Clean title by removing date/extension if formatted
        title = file.stem
        # If title matches name_YYYY-MM-DD pattern, extract just the title
        if "_" in title:
            parts = title.split("_")
            # If last part matches YYYY-MM-DD date format
            if len(parts) > 1 and len(parts[-1]) == 10 and parts[-1][4] == "-" and parts[-1][7] == "-":
                title = " ".join(parts[:-1])
            else:
                title = title.replace("_", " ")

        transcripts.append({
            "filename": file.name,
            "title": title,
            "date": mtime,
            "size_kb": round(stat.st_size / 1024, 2)
        })
        
    # Sort by date descending (newest first)
    transcripts.sort(key=lambda x: x["date"], reverse=True)
    return {"transcripts": transcripts}


@app.get("/api/transcripts/{filename}")
async def get_saved_transcript(filename: str):
    """Retrieve content of a specific transcript file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")
    
    try:
        text = file_path.read_text(encoding="utf-8")
        return {"filename": filename, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


# ---------------------------------------------------------------------------

# Report Download Routes
# ---------------------------------------------------------------------------

@app.get("/api/reports/pdf/{filename}")
async def download_pdf_report(filename: str):
    """Retrieve and download a generated PDF meeting report."""
    pdf_file_path = Path(__file__).resolve().parent / "reports" / "pdf" / filename
    if not pdf_file_path.exists():
        raise HTTPException(status_code=404, detail="PDF report file not found.")
    return FileResponse(pdf_file_path, media_type="application/pdf", filename=filename)


@app.get("/api/reports/excel/{filename}")
async def download_excel_report(filename: str):
    """Retrieve and download a generated Excel action tracker."""
    excel_file_path = Path(__file__).resolve().parent / "reports" / "excel" / filename
    if not excel_file_path.exists():
        raise HTTPException(status_code=404, detail="Excel action tracker file not found.")
    return FileResponse(
        excel_file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )

@app.get("/api/reports/word/{filename}")
async def download_word_report(filename: str):
    """Retrieve and download a generated Word meeting report."""
    word_file_path = Path(__file__).resolve().parent / "reports" / "word" / filename
    if not word_file_path.exists():
        raise HTTPException(status_code=404, detail="Word report file not found.")
    return FileResponse(
        word_file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


def main():
    """Start the FastAPI backend with uvicorn."""
    logger.info("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("app.py:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
