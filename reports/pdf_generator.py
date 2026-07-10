"""PDF Report Generator using xhtml2pdf.

Renders meeting summary data into a professional corporate PDF report
containing all MoM sections: Discussion Points, Pending Items, Parking Lot, etc.
"""

import time
from datetime import datetime
from html import escape
from pathlib import Path

from xhtml2pdf import pisa

from config.settings import (
    COMPANY_LOGO_PATH,
    COMPANY_NAME,
    COMPANY_SECONDARY_COLOR,
    COMPANY_THEME_COLOR,
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

        template_html_path = self._template_dir / "meeting_template.html"
        styles_css_path = self._template_dir / "styles.css"

        if not template_html_path.exists() or not styles_css_path.exists():
            raise FileNotFoundError("PDF templates (HTML or CSS) are missing.")

        html_template = template_html_path.read_text(encoding="utf-8")
        css_template = styles_css_path.read_text(encoding="utf-8")

        styled_css = css_template.replace("{company_name}", COMPANY_NAME)
        styled_css = styled_css.replace("{theme_color}", COMPANY_THEME_COLOR)
        styled_css = styled_css.replace("{secondary_color}", COMPANY_SECONDARY_COLOR)

        # ── Resolve attendees (manual list takes precedence) ──────────────────
        attendees = summary_data.get("attendees") or []
        participants = summary_data.get("participants", [])
        display_attendees = attendees if attendees else participants

        action_items = summary_data.get("action_items", [])
        discussion_points = summary_data.get("discussion_points", [])

        topics_html = self._to_bullet_list(summary_data.get("topics", []))
        decisions_html = self._to_bullet_list(summary_data.get("decisions", []))

        attendees_html = "".join(
            f"<li>{self._html(a)}</li>" for a in display_attendees
        ) or "<li>None detected</li>"

        action_items_html = self._build_action_items_rows(action_items)
        discussion_points_html = self._build_discussion_points_html(discussion_points)
        generated_at = self._format_datetime(summary_data.get("generated_at", ""))
        logo_path = COMPANY_LOGO_PATH
        if not Path(logo_path).exists():
            logo_path = str(Path(__file__).resolve().parent / "assets" / "company_logo.png")

        formatted_html = html_template.format(
            styles=styled_css,
            logo_path=logo_path,
            company_name=self._html(COMPANY_NAME),
            meeting_title=self._html(summary_data.get("meeting_title", "Meeting")),
            meeting_date=self._html(summary_data.get("meeting_date", "")),
            meeting_type=self._html(summary_data.get("meeting_type", "General")),
            generated_at=self._html(generated_at),
            executive_summary=self._html(summary_data.get("executive_summary", "")),
            attendees_html=attendees_html,
            topics_html=topics_html,
            decisions_html=decisions_html,
            action_items_rows=action_items_html,
            discussion_points_html=discussion_points_html,
        )

        logger.info("Compiling HTML to PDF: %s", output_path.name)
        with open(output_path, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(
                formatted_html,
                dest=pdf_file,
            )

        if pisa_status.err:
            raise RuntimeError(f"xhtml2pdf compilation failed with error code {pisa_status.err}")

        duration = time.time() - start_time
        file_size = output_path.stat().st_size / 1024
        logger.info(
            "PDF generation complete. Path=%s, Size=%.2f KB, Duration=%.2fs",
            output_path,
            file_size,
            duration,
        )
        return output_path

    @staticmethod
    def _to_bullet_list(items: list) -> str:
        """Convert a list of strings into a styled HTML bullet list."""
        if not items:
            return "<p class='paragraph' style='color:#718096; font-style:italic;'>None detected.</p>"
        rows = "\n".join(f"<li>{PDFGenerator._html(item)}</li>" for item in items)
        return f"<ul class='bullet-list'>\n{rows}\n</ul>"

    @staticmethod
    def _build_discussion_points_html(discussion_points: list) -> str:
        """Build HTML cards for each discussion point with all 12 MoM fields."""
        if not discussion_points:
            return "<p class='paragraph' style='color:#718096; font-style:italic;'>No discussion points captured.</p>"

        cards = []
        for i, dp in enumerate(discussion_points, start=1):
            if isinstance(dp, dict):
                get = lambda k, default="": dp.get(k, default)  # noqa: E731
            else:
                get = lambda k, default="": getattr(dp, k, default)  # noqa: E731

            point = get("point", f"Discussion {i}")
            detailed_summary = get("detailed_summary", "")
            decision = get("decision", "No Decision Taken")
            task = get("task", "No Action Item")
            assigned_to = get("assigned_to", "Not Specified")
            deadline = get("deadline", "Not Specified")
            priority = get("priority", "Medium")
            status = get("status", "Open")
            risks_or_concerns = get("risks_or_concerns", "")
            suggestions = get("suggestions", "")
            follow_up_required = get("follow_up_required", "No")
            notes = get("notes", "")

            card = f"""
<div class='dp-card'>
    <div class='dp-heading'>{PDFGenerator._html(f'{i}. {point}')}</div>
    <table class='dp-table'>
        <tr>
            <td class='dp-label'>Summary</td>
            <td class='dp-value' colspan='3'>{PDFGenerator._html(detailed_summary)}</td>
        </tr>
        <tr>
            <td class='dp-label'>Decision</td>
            <td class='dp-value' colspan='3'>{PDFGenerator._html(decision)}</td>
        </tr>
        <tr>
            <td class='dp-label'>Task / Action Item</td>
            <td class='dp-value' colspan='3'>{PDFGenerator._html(task)}</td>
        </tr>
        <tr>
            <td class='dp-label'>Assigned To</td>
            <td class='dp-value'>{PDFGenerator._html(assigned_to)}</td>
            <td class='dp-label'>Deadline</td>
            <td class='dp-value'>{PDFGenerator._html(deadline)}</td>
        </tr>
    </table>
</div>"""
            cards.append(card)

        return "\n".join(cards)

    @staticmethod
    def _build_action_items_rows(action_items: list) -> str:
        """Build HTML table rows for action items."""
        if not action_items:
            return "<tr><td colspan='6' align='center' style='color:#718096; font-style:italic;'>No details available.</td></tr>"

        rows = []
        for item in action_items:
            task = item.get("task", "") if isinstance(item, dict) else getattr(item, "task", "")
            owner = item.get("owner", "") if isinstance(item, dict) else getattr(item, "owner", "")
            target_date = item.get("target_date", "") if isinstance(item, dict) else getattr(item, "target_date", "")
            priority = item.get("priority", "Medium") if isinstance(item, dict) else getattr(item, "priority", "Medium")
            status = item.get("status", "Pending") if isinstance(item, dict) else getattr(item, "status", "Pending")
            notes = item.get("notes", "") if isinstance(item, dict) else getattr(item, "notes", "")

            is_task = False
            if owner and owner.strip():
                owner_lower = owner.strip().lower()
                if owner_lower not in (
                    "general",
                    "none",
                    "n/a",
                    "nil",
                    "general discussion",
                    "information",
                    "info",
                ):
                    is_task = True
            if status and status.strip().lower() in ("information", "info"):
                is_task = False

            if is_task:
                participants_display = owner.strip()
                target_date_display = target_date
                priority_display = priority
                status_display = status
                row_class = ""
            else:
                participants_display = "Information"
                target_date_display = ""
                priority_display = ""
                status_display = "Information"
                row_class = "class='muted-row'"

            rows.append(
                f"""
            <tr {row_class}>
                <td width="34%">{PDFGenerator._html(task)}</td>
                <td width="16%">{PDFGenerator._html(participants_display)}</td>
                <td width="14%">{PDFGenerator._html(target_date_display)}</td>
                <td width="10%">{PDFGenerator._badge(priority_display)}</td>
                <td width="12%">{PDFGenerator._badge(status_display)}</td>
                <td width="14%">{PDFGenerator._html(notes)}</td>
            </tr>
            """
            )
        return "\n".join(rows)

    @staticmethod
    def _html(value: object) -> str:
        """Escape untrusted LLM/user content before injecting it into the report HTML."""
        return escape(str(value or ""), quote=True)

    @staticmethod
    def _css_token(value: object) -> str:
        token = str(value or "").strip().lower().replace(" ", "-")
        return "".join(char for char in token if char.isalnum() or char == "-") or "neutral"

    @staticmethod
    def _badge(value: str) -> str:
        if not value:
            return ""
        safe_value = PDFGenerator._html(value)
        return f"<div class='pill pill-{PDFGenerator._css_token(value)}'>{safe_value}</div>"

    @staticmethod
    def _format_datetime(value: str) -> str:
        if not value:
            return ""
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
        return parsed.strftime("%d %b %Y, %I:%M %p")
