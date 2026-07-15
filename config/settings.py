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

# LLM Reliability & Chunking settings
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_INITIAL_BACKOFF: float = float(os.getenv("LLM_INITIAL_BACKOFF", "2.0"))
LLM_BACKOFF_FACTOR: float = float(os.getenv("LLM_BACKOFF_FACTOR", "2.0"))
LLM_SAFETY_MARGIN: float = float(os.getenv("LLM_SAFETY_MARGIN", "0.9"))
LLM_CHUNK_SIZE_TOKENS: int = int(os.getenv("LLM_CHUNK_SIZE_TOKENS", "900"))
LLM_THROTTLE_DELAY: float = float(os.getenv("LLM_THROTTLE_DELAY", "2.0"))
LLM_REQUEST_THROTTLE_SECONDS: float = float(os.getenv("LLM_REQUEST_THROTTLE_SECONDS", "0"))
NVIDIA_REQUEST_THROTTLE_SECONDS: float = float(os.getenv("NVIDIA_REQUEST_THROTTLE_SECONDS", "30"))
LLM_CHUNK_MIN_TOKENS: int = int(os.getenv("LLM_CHUNK_MIN_TOKENS", "700"))
GROQ_RPM_LIMIT: int = int(os.getenv("GROQ_RPM_LIMIT", "30"))
GROQ_TPM_LIMIT: int = int(os.getenv("GROQ_TPM_LIMIT", "12000"))

# Pipeline-specific model overrides
CHUNK_EXTRACTOR_MODEL: str = os.getenv("CHUNK_EXTRACTOR_MODEL", "openai/gpt-oss-120b")
NVIDIA_MOM_MODEL: str = os.getenv("NVIDIA_MOM_MODEL", "z-ai/glm-5.2")
AGENT1_MODEL: str = os.getenv("AGENT1_MODEL", "openai/gpt-oss-120b")
AGENT2_MODEL: str = os.getenv("AGENT2_MODEL", "deepseek-ai/deepseek-v4-flash")
AGENT3_MODEL: str = os.getenv("AGENT3_MODEL", "z-ai/glm-5.2")
AGENT4_MODEL: str = os.getenv("AGENT4_MODEL", "z-ai/glm-5.2")
AGENT5_MODEL: str = os.getenv("AGENT5_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
AGENT6_MODEL: str = os.getenv("AGENT6_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Phase 4: Report Generation Settings
# ---------------------------------------------------------------------------
COMPANY_NAME: str = os.getenv("COMPANY_NAME", "VAPS TECHNOSOFT PVT. LTD.")
COMPANY_LOGO_PATH: str = os.getenv("COMPANY_LOGO_PATH", r"C:\Users\Vaps\PycharmProjects\AIMOM\reports\assets\company_logo.png")
COMPANY_THEME_COLOR: str = os.getenv("COMPANY_THEME_COLOR", "#1e3a8a")  # Deep Blue
COMPANY_SECONDARY_COLOR: str = os.getenv("COMPANY_SECONDARY_COLOR", "#3b82f6")  # Light Blue
