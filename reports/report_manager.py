"""Report Manager — coordinates PDF and Excel document exports.

Validates summary details and triggers file writers.
"""

import time
from pathlib import Path
from typing import Dict, Union

from ai.models.meeting import MeetingSummary
from reports.pdf_generator import PDFGenerator
from reports.excel_generator import ExcelGenerator
from reports.word_generator import WordGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportValidationError(Exception):
    """Raised when meeting summary data fails validation checks before exporting."""


class ReportManager:
    """Orchestrates document exports, handling validation, naming conventions, and file writes."""

    def __init__(self) -> None:
        self._pdf_generator = PDFGenerator()
        self._excel_generator = ExcelGenerator()
        self._word_generator = WordGenerator()
        self._pdf_dir = Path(__file__).resolve().parent / "pdf"
        self._excel_dir = Path(__file__).resolve().parent / "excel"
        self._word_dir = Path(__file__).resolve().parent / "word"

        self._pdf_dir.mkdir(parents=True, exist_ok=True)
        self._excel_dir.mkdir(parents=True, exist_ok=True)
        self._word_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, summary: Union[dict, MeetingSummary]) -> Dict[str, str]:
        """Validate summary data and generate both PDF and Excel reports.

        Args:
            summary: MeetingSummary Pydantic model or raw dictionary.

        Returns:
            Dictionary containing 'pdf' and 'excel' paths to generated files.

        Raises:
            ReportValidationError: On missing required fields.
        """
        start_time = time.time()
        
        # Convert Pydantic model to dictionary if necessary
        summary_data = summary.model_dump() if isinstance(summary, MeetingSummary) else summary

        # 1. Validation Checks
        self._validate_summary_data(summary_data)

        # 2. Establish Names
        title = summary_data.get("meeting_title", "Meeting")
        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
        safe_title = safe_title.strip().replace(" ", "_") or "Meeting"
        
        date_str = summary_data.get("meeting_date", "")
        # Fallback date if empty
        if not date_str:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")

        pdf_filename = f"{safe_title}_{date_str}.pdf"
        excel_filename = f"{safe_title}_{date_str}.xlsx"
        word_filename = f"{safe_title}_{date_str}.docx"

        logger.info("Starting report export generation for title: '%s'", title)

        # 3. Generate PDF
        try:
            pdf_path = self._pdf_generator.generate(summary_data, pdf_filename)
        except Exception as exc:
            logger.exception("Failed to generate PDF report")
            raise RuntimeError(f"PDF generation failed: {exc}") from exc

        # 4. Generate Excel
        try:
            excel_path = self._excel_generator.generate(summary_data, excel_filename)
        except Exception as exc:
            logger.exception("Failed to generate Excel action tracker")
            raise RuntimeError(f"Excel generation failed: {exc}") from exc

        # 5. Generate Word
        word_path = None
        try:
            word_path = self._word_generator.generate(summary_data, word_filename)
        except Exception as exc:
            logger.warning("Word generation failed (non-critical): %s", exc)

        total_duration = time.time() - start_time
        logger.info(
            "Report exports completed successfully in %.2fs. PDF: %s, Excel: %s, Word: %s",
            total_duration,
            pdf_path.name,
            excel_path.name,
            word_path.name if word_path else "skipped",
        )

        result = {
            "pdf": str(pdf_path.resolve()),
            "excel": str(excel_path.resolve()),
        }
        if word_path:
            result["word"] = str(word_path.resolve())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_summary_data(data: dict) -> None:
        """Enforce presence of essential fields prior to document compilation."""
        required_fields = ["meeting_title", "executive_summary"]
        missing = [f for f in required_fields if not data.get(f)]

        if missing:
            msg = f"Cannot generate reports. Missing required fields: {', '.join(missing)}"
            logger.error(msg)
            raise ReportValidationError(msg)

        # Log structure details for trace purposes
        logger.debug(
            "Validation successful. Title: '%s', Topics count: %d, Action items count: %d",
            data.get("meeting_title"),
            len(data.get("topics", [])),
            len(data.get("action_items", [])),
        )
