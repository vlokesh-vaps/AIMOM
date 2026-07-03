"""PDF Report Generator using xhtml2pdf.

Renders meeting summary data into a professional corporate PDF report.
"""

import time
from pathlib import Path
from xhtml2pdf import pisa

from config.settings import (
    COMPANY_NAME,
    COMPANY_LOGO_PATH,
    COMPANY_THEME_COLOR,
    COMPANY_SECONDARY_COLOR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFGenerator:
    """PDF generator converting structured meeting summary data to HTML and printing it via xhtml2pdf."""

    def __init__(self) -> None:
        self._template_dir = Path(__file__).resolve().parent / "templates"
        self._output_dir = Path(__file__).resolve().parent / "pdf"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, summary_data: dict, filename: str) -> Path:
        """Generate a styled corporate PDF report from meeting intelligence.

        Args:
            summary_data: Dictionary conforming to MeetingSummary schema.
            filename: Output filename (e.g. Review_Meeting_2026-06-30.pdf).

        Returns:
            Path to the saved PDF file.
        """
        start_time = time.time()
        output_path = self._output_dir / filename

        # Read template files
        template_html_path = self._template_dir / "meeting_template.html"
        styles_css_path = self._template_dir / "styles.css"

        if not template_html_path.exists() or not styles_css_path.exists():
            raise FileNotFoundError("PDF templates (HTML or CSS) are missing.")

        html_template = template_html_path.read_text(encoding="utf-8")
        css_template = styles_css_path.read_text(encoding="utf-8")

        # 1. Format CSS styles with branding variables via direct replace
        styled_css = css_template.replace("{company_name}", COMPANY_NAME)
        styled_css = styled_css.replace("{theme_color}", COMPANY_THEME_COLOR)
        styled_css = styled_css.replace("{secondary_color}", COMPANY_SECONDARY_COLOR)


        # 2. Format HTML content lists
        topics_html = self._to_bullet_list(summary_data.get("topics", []))
        decisions_html = self._to_bullet_list(summary_data.get("decisions", []))
        risks_html = self._to_bullet_list(summary_data.get("risks", []))
        questions_html = self._to_bullet_list(summary_data.get("questions", []))
        followups_html = self._to_bullet_list(summary_data.get("followups", []))
        
        participants_html = "".join(
            f"<li>{p}</li>" for p in summary_data.get("participants", [])
        ) or "<li>None detected</li>"

        keywords_html = "".join(
            f"<li>{k}</li>" for k in summary_data.get("keywords", [])
        ) or "<li>None</li>"

        action_items_html = self._build_action_items_rows(summary_data.get("action_items", []))
        timeline_html = self._build_timeline_html(summary_data.get("timeline", []))

        # Check logo path existence for local filesystem resolution
        logo_path = COMPANY_LOGO_PATH
        if not Path(logo_path).exists():
            # Fallback to absolute assets path in reports directory
            logo_path = str(Path(__file__).resolve().parent / "assets" / "company_logo.png")

        # 3. Inject variables into the HTML template
        formatted_html = html_template.format(
            styles=styled_css,
            logo_path=logo_path,
            company_name=COMPANY_NAME,
            meeting_title=summary_data.get("meeting_title", "Meeting"),
            meeting_date=summary_data.get("meeting_date", ""),
            meeting_duration=summary_data.get("meeting_duration", "Unknown"),
            meeting_type=summary_data.get("meeting_type", "General"),
            generated_at=summary_data.get("generated_at", ""),
            overall_sentiment=summary_data.get("overall_sentiment", "Neutral"),
            overall_sentiment_lower=summary_data.get("overall_sentiment", "Neutral").lower(),
            executive_summary=summary_data.get("executive_summary", ""),
            participants_html=participants_html,
            topics_html=topics_html,
            decisions_html=decisions_html,
            action_items_rows=action_items_html,
            risks_html=risks_html,
            questions_html=questions_html,
            timeline_html=timeline_html,
            keywords_html=keywords_html,
            followups_html=followups_html,
        )

        # 4. Compile HTML to PDF via xhtml2pdf
        logger.info("Compiling HTML to PDF: %s", output_path.name)
        with open(output_path, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(
                formatted_html,
                dest=pdf_file,
            )

        if pisa_status.err:
            raise RuntimeError(f"xhtml2pdf compilation failed with error code {pisa_status.err}")

        # Metrics logging
        duration = time.time() - start_time
        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(
            "PDF generation complete. Path=%s, Size=%.2f KB, Duration=%.2fs",
            output_path,
            file_size,
            duration,
        )
        return output_path

    # ------------------------------------------------------------------
    # Format Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bullet_list(items: list) -> str:
        """Convert a list of strings into a styled HTML bullet list."""
        if not items:
            return "<p class='paragraph' style='color:#718096; font-style:italic;'>None detected.</p>"
        return "<ul class='bullet-list'>\n" + "\n".join(f"<li>{item}</li>" for item in items) + "\n</ul>"

    @staticmethod
    def _build_action_items_rows(action_items: list) -> str:
        """Build HTML table rows for action items."""
        if not action_items:
            return "<tr><td colspan='6' align='center' style='color:#718096; font-style:italic;'>No details available.</td></tr>"

        rows = []
        for item in action_items:
            # Handle if dict or pydantic model
            task = item.get("task", "") if isinstance(item, dict) else getattr(item, "task", "")
            owner = item.get("owner", "") if isinstance(item, dict) else getattr(item, "owner", "")
            target_date = item.get("target_date", "") if isinstance(item, dict) else getattr(item, "target_date", "")
            priority = item.get("priority", "Medium") if isinstance(item, dict) else getattr(item, "priority", "Medium")
            status = item.get("status", "Pending") if isinstance(item, dict) else getattr(item, "status", "Pending")
            notes = item.get("notes", "") if isinstance(item, dict) else getattr(item, "notes", "")

            # Check if it's a task or general discussion
            is_task = False
            if owner and owner.strip():
                o_lower = owner.strip().lower()
                if o_lower not in ("unknown", "general", "none", "n/a", "nil", "general discussion", "information", "info"):
                    is_task = True
            if status and status.strip().lower() in ("information", "info"):
                is_task = False

            if is_task:
                participants_display = owner.strip()
                target_date_display = target_date
                priority_display = priority
                status_display = status
                row_style = ""
            else:
                participants_display = "Information"
                target_date_display = ""
                priority_display = ""
                status_display = "Information"
                # Apply a soft style to general discussion rows
                row_style = "style='background-color: #fafafa; font-style: italic; color: #4a5568;'"

            rows.append(f"""
            <tr {row_style}>
                <td>{task}</td>
                <td>{participants_display}</td>
                <td>{target_date_display}</td>
                <td>{priority_display}</td>
                <td>{status_display}</td>
                <td>{notes}</td>
            </tr>
            """)
        return "\n".join(rows)

    @staticmethod
    def _build_timeline_html(timeline: list) -> str:
        """Build HTML table rows for chronological timeline events."""
        if not timeline:
            return "<tr><td colspan='2' align='center' style='color:#718096; font-style:italic;'>No timeline details available.</td></tr>"

        rows = []
        for event in timeline:
            if " - " in event:
                time_str, event_str = event.split(" - ", 1)
            elif ": " in event:
                time_str, event_str = event.split(": ", 1)
            else:
                time_str, event_str = "•", event

            rows.append(f"""
            <tr>
                <td class="timeline-time">{time_str.strip()}</td>
                <td class="timeline-event">{event_str.strip()}</td>
            </tr>
            """)
        return "\n".join(rows)
