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

        attendees_list = ", ".join(self._html(a) for a in display_attendees) if display_attendees else "None"
        unified_table_rows = self._build_unified_table(discussion_points, action_items)

        formatted_html = html_template.format(
            styles=styled_css,
            company_name=self._html(COMPANY_NAME),
            meeting_title=self._html(summary_data.get("meeting_title", "Meeting")),
            meeting_date=self._html(summary_data.get("meeting_date", "")),
            chaired_by=self._html(summary_data.get("chaired_by", "") or "Not specified"),
            meeting_type=self._html(summary_data.get("meeting_type", "General")),
            organization=self._html(summary_data.get("organization", "") or COMPANY_NAME),
            attendees_list=attendees_list,
            absents=self._html(summary_data.get("absents", "") or "Nil"),
            unified_table_rows=unified_table_rows,
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
    def _build_unified_table(discussion_points: list, action_items: list) -> str:
        """Build the unified details table combining discussions and actions grouped by agenda."""
        actions_by_agenda: dict[str, list] = {}
        for item in action_items:
            agenda = PDFGenerator._get(item, "agenda_item", "").strip() or "Off Agenda Discussion"
            actions_by_agenda.setdefault(agenda, []).append(item)

        rows = []
        used_action_ids: set[int] = set()

        for idx, dp in enumerate(discussion_points, start=1):
            agenda = PDFGenerator._get(dp, "agenda_item", "Off Agenda Discussion").strip() or "Off Agenda Discussion"
            matched = actions_by_agenda.get(agenda, [])
            actions_html = PDFGenerator._join_list_html(PDFGenerator._get(item, "task", "") for item in matched) or "No Action Item"
            assigned_html = PDFGenerator._html(PDFGenerator._join_unique(PDFGenerator._get(item, "owner", "") for item in matched) or PDFGenerator._get(dp, "assigned_to", "Not Specified"))
            date_html = PDFGenerator._html(PDFGenerator._join_unique(PDFGenerator._get(item, "target_date", "") for item in matched) or PDFGenerator._get(dp, "deadline", "Not Specified"))

            summary_html = PDFGenerator._html(PDFGenerator._get(dp, "point", ""))
            details = PDFGenerator._get(dp, "detailed_summary", "")
            if details and details != PDFGenerator._get(dp, "point", ""):
                summary_html += f"<br/>{PDFGenerator._html(details)}"
            decision = PDFGenerator._get(dp, "decision", "")
            if decision and decision != "No Decision Taken":
                summary_html += f"<br/><em>Decision:</em> {PDFGenerator._html(decision)}"

            rows.append(
                f"""
                <tr>
                    <td>{PDFGenerator._html(f"{idx}. {agenda}")}</td>
                    <td>{actions_html}</td>
                    <td>{summary_html}</td>
                    <td>{assigned_html}</td>
                    <td>{date_html}</td>
                </tr>
                """
            )
            for item in matched:
                used_action_ids.add(id(item))

        next_index = len(rows) + 1
        for item in action_items:
            if id(item) in used_action_ids:
                continue
            agenda = PDFGenerator._get(item, "agenda_item", "").strip() or "Off Agenda Discussion"
            rows.append(
                f"""
                <tr>
                    <td>{PDFGenerator._html(f"{next_index}. {agenda}")}</td>
                    <td>{PDFGenerator._html(PDFGenerator._get(item, "task", ""))}</td>
                    <td></td>
                    <td>{PDFGenerator._html(PDFGenerator._get(item, "owner", ""))}</td>
                    <td>{PDFGenerator._html(PDFGenerator._get(item, "target_date", ""))}</td>
                </tr>
                """
            )
            next_index += 1

        if not rows:
            return "<tr><td colspan='5' align='center'>No detailed records found.</td></tr>"
        return "\n".join(rows)

    @staticmethod
    def _join_list_html(values) -> str:
        items = [PDFGenerator._html(str(value).strip()) for value in values if str(value or "").strip()]
        if not items:
            return ""
        return "<ul class='action-item-list'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    @staticmethod
    def _join_unique(values) -> str:
        seen: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in ("Not Specified", "-"):
                continue
            if text not in seen:
                seen.append(text)
        return " / ".join(seen)

    @staticmethod
    def _get(item, key: str, default: str = "") -> str:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

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
