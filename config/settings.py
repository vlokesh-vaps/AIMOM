"""Application settings and configuration.

Loads environment variables from .env and exposes constants
used throughout the application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000
CHANNELS: int = 1
AUDIO_DTYPE: str = "int16"

SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".mp4",
)

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
RECORDINGS_DIR: Path = PROJECT_ROOT / "recordings"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
TEMP_DIR: Path = PROJECT_ROOT / "temp"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# NVIDIA NIM settings
# ---------------------------------------------------------------------------
NVIDIA_GRPC_SERVER: str = "grpc.nvcf.nvidia.com:443"

NVIDIA_MODEL_MAP: dict[str, dict[str, str]] = {
    "NVIDIA Parakeet CTC 1.1B": {
        "function_id": "1598d209-5e27-4d3c-8079-4751568b1081",
        "mode": "streaming",
    },
    "NVIDIA Whisper Large v3": {
        "function_id": "b702f636-f60c-4a3d-a6f4-f3568c13bd7d",
        "mode": "offline",
    },
}

# ---------------------------------------------------------------------------
# Deepgram settings
# ---------------------------------------------------------------------------
DEEPGRAM_MODEL_MAP: dict[str, str] = {
    "Deepgram Nova-3": "nova-3",
    "Deepgram Nova-2": "nova-2",
}

# ---------------------------------------------------------------------------
# Language mappings
# ---------------------------------------------------------------------------
LANGUAGE_OPTIONS: list[str] = [
    "English",
    "Kannada",
    "Hindi",
    "Tamil",
    "Telugu",
    "Auto",
]

NVIDIA_LANGUAGE_MAP: dict[str, str] = {
    "English": "en-US",
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Auto": "en-US",
}

DEEPGRAM_LANGUAGE_MAP: dict[str, str | None] = {
    "English": "en",
    "Kannada": "kn",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Auto": None,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT: int = 3

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT: int = 300  # seconds

# ---------------------------------------------------------------------------
# Phase 3: AI Intelligence Module Settings
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Defaults for LLM Analysis
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "nvidia")  # nvidia, groq, gemini
AI_MODEL: str = os.getenv("AI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
AI_TEMPERATURE: float = float(os.getenv("AI_TEMPERATURE", "0.1"))
AI_MAX_TOKENS: int = int(os.getenv("AI_MAX_TOKENS", "4096"))
AI_TOP_P: float = float(os.getenv("AI_TOP_P", "1.0"))
AI_TIMEOUT: int = int(os.getenv("AI_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:latest")

# ---------------------------------------------------------------------------
# LLM Reliability & Chunking settings
# ---------------------------------------------------------------------------
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_INITIAL_BACKOFF: float = float(os.getenv("LLM_INITIAL_BACKOFF", "2.0"))
LLM_BACKOFF_FACTOR: float = float(os.getenv("LLM_BACKOFF_FACTOR", "2.0"))
LLM_CHUNK_SIZE_TOKENS: int = int(os.getenv("LLM_CHUNK_SIZE_TOKENS", "900"))

# ---------------------------------------------------------------------------
# ProviderManager — failover & retry settings
# ---------------------------------------------------------------------------
PROVIDER_MAX_RETRIES: int = int(os.getenv("PROVIDER_MAX_RETRIES", "3"))
PROVIDER_INITIAL_BACKOFF: float = float(os.getenv("PROVIDER_INITIAL_BACKOFF", "2.0"))
PROVIDER_BACKOFF_FACTOR: float = float(os.getenv("PROVIDER_BACKOFF_FACTOR", "2.0"))
PROVIDER_HEALTH_CHECK_INTERVAL: float = float(os.getenv("PROVIDER_HEALTH_CHECK_INTERVAL", "60"))
PROVIDER_COOLDOWN_SECONDS: float = float(os.getenv("PROVIDER_COOLDOWN_SECONDS", "30.0"))

# ---------------------------------------------------------------------------
# Async Extraction Pipeline settings
# ---------------------------------------------------------------------------
# Max chunks processed concurrently (bounded by asyncio.Semaphore)
MAX_CONCURRENT_EXTRACTIONS: int = int(os.getenv("MAX_CONCURRENT_EXTRACTIONS", "3"))
# Model used for per-chunk structured extraction (NVIDIA primary)
EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "deepseek-ai/deepseek-v4-flash")
# Model used for final business-language synthesis (NVIDIA primary)
SYNTHESIS_MODEL: str = os.getenv("SYNTHESIS_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# Max output tokens for extraction (compact output to avoid truncation)
EXTRACTION_MAX_TOKENS: int = int(os.getenv("EXTRACTION_MAX_TOKENS", "2048"))
# Max output tokens for final synthesis
SYNTHESIS_MAX_TOKENS: int = int(os.getenv("SYNTHESIS_MAX_TOKENS", "3072"))
# Prompt version tag stored in checkpoints for invalidation
PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v2.0")
# Checkpoint directory
CHECKPOINT_DIR: Path = TEMP_DIR / "checkpoints"

# ---------------------------------------------------------------------------
# 4-Agent pipeline model assignments
# ---------------------------------------------------------------------------
# Agent 1: Topic Segmentation (NVIDIA DeepSeek)
AGENT1_MODEL: str = os.getenv("AGENT1_MODEL", "deepseek-ai/deepseek-v4-flash")
# Agent 2: Discussion + Action Extraction (NVIDIA GLM)
AGENT2_MODEL: str = os.getenv("AGENT2_MODEL", "z-ai/glm-5.2")
# Agent 3: Final Synthesis (NVIDIA Nemotron)
AGENT3_MODEL: str = os.getenv("AGENT3_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# Agent 4: Validation (Groq — optional, never blocks)
AGENT4_MODEL: str = os.getenv("AGENT4_MODEL", "openai/gpt-oss-120b")
# Default NVIDIA model for non-agent calls (translation, etc.)
NVIDIA_MOM_MODEL: str = os.getenv("NVIDIA_MOM_MODEL", "z-ai/glm-5.2")
# Groq fallback model used by ProviderManager when NVIDIA is unavailable
GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Phase 4: Report Generation Settings
# ---------------------------------------------------------------------------
COMPANY_NAME: str = os.getenv("COMPANY_NAME", "VAPS TECHNOSOFT PVT. LTD.")
COMPANY_LOGO_PATH: str = os.getenv("COMPANY_LOGO_PATH", r"C:\Users\Vaps\PycharmProjects\AIMOM\reports\assets\company_logo.png")
COMPANY_THEME_COLOR: str = os.getenv("COMPANY_THEME_COLOR", "#1e3a8a")  # Deep Blue
COMPANY_SECONDARY_COLOR: str = os.getenv("COMPANY_SECONDARY_COLOR", "#3b82f6")  # Light Blue
