"""Excel Action Tracker Generator using openpyxl.

Outputs a multi-sheet Excel workbook:
  Sheet 1 — Action Tracker (all tasks with priority, status, deadline)
  Sheet 2 — Discussion Points (full detail for every discussion point)
  Sheet 3 — Pending Items & Parking Lot
"""

import time
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from config.settings import COMPANY_THEME_COLOR
from utils.logger import get_logger

logger = get_logger(__name__)


class ExcelGenerator:
    """Excel generator creating stylised corporate MoM workbooks from meeting summary data."""

    def __init__(self) -> None:
        self._output_dir = Path(__file__).resolve().parent / "excel"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, summary_data: dict, filename: str) -> Path:
        """Generate a styled corporate Excel workbook (3 sheets).

        Args:
            summary_data: Dictionary conforming to MeetingSummary schema.
            filename: Output filename (e.g. Review_Meeting_2026-06-30.xlsx).

        Returns:
            Path to the saved Excel workbook.
        """
        start_time = time.time()
        output_path = self._output_dir / filename

        wb = Workbook()

        theme_hex = COMPANY_THEME_COLOR.lstrip("#")
        if len(theme_hex) != 6:
            theme_hex = "1E3A8A"

        # Build shared styles
        styles = self._make_styles(theme_hex)

        # Sheet 1 — Action Tracker
        ws1 = wb.active
        ws1.title = "Action Tracker"
        self._build_action_tracker(ws1, summary_data, styles)

        # Sheet 2 — Discussion Points
        ws2 = wb.create_sheet("Discussion Points")
        self._build_discussion_points(ws2, summary_data, styles)

        # Sheet 3 — Pending Items & Parking Lot
        ws3 = wb.create_sheet("Pending & Parking Lot")
        self._build_pending_parking(ws3, summary_data, styles)

        wb.save(str(output_path))

        duration = time.time() - start_time
        file_size = output_path.stat().st_size / 1024
        logger.info(
            "Excel generation complete. Path=%s, Size=%.2f KB, Duration=%.2fs",
            output_path,
            file_size,
            duration,
        )
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # Sheet builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_action_tracker(self, ws, summary_data: dict, styles: dict) -> None:
        """Sheet 1: Action Tracker with enhanced columns."""
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Meeting",
            "Task / Action Plan",
            "Assigned To",
            "Target Date",
            "Priority",
            "Status",
            "Risks / Concerns",
            "Notes",
        ]
        self._write_headers(ws, headers, styles)

        action_items = summary_data.get("action_items", [])
        meeting_title = summary_data.get("meeting_title", "Meeting")

        if not action_items:
            row = [meeting_title, "No action items identified.", "", "", "", "", "", ""]
            self._write_data_row(ws, 2, row, styles)
        else:
            for row_idx, item in enumerate(action_items, start=2):
                task = self._get(item, "task", "")
                owner = self._get(item, "owner", "")
                target_date = self._get(item, "target_date", "")
                priority = self._get(item, "priority", "Medium")
                status = self._get(item, "status", "Pending")
                notes = self._get(item, "notes", "")

                is_task = self._is_task(owner, status)
                assignee = owner.strip() if is_task else "—"
                date_val = target_date if is_task else ""
                status_val = status if is_task else "Information"

                row = [
                    meeting_title,
                    task,
                    assignee,
                    date_val,
                    priority if is_task else "",
                    status_val,
                    "",   # risks not in ActionItem schema, left blank
                    notes,
                ]
                self._write_data_row(ws, row_idx, row, styles)

        last_row = max(len(action_items) + 1, 2)
        ws.auto_filter.ref = f"A1:H{last_row}"
        ws.freeze_panes = "A2"
        self._autofit(ws)

    def _build_discussion_points(self, ws, summary_data: dict, styles: dict) -> None:
        """Sheet 2: Full Detail Discussion Points (one row per discussion point)."""
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "#",
            "Discussion Point",
            "Detailed Summary",
            "Decision",
            "Task / Action Item",
            "Assigned To",
            "Deadline",
            "Priority",
            "Status",
            "Risks / Concerns",
            "Suggestions",
            "Follow-up Required",
            "Notes",
        ]
        self._write_headers(ws, headers, styles)

        discussion_points = summary_data.get("discussion_points", [])
        meeting_title = summary_data.get("meeting_title", "Meeting")

        if not discussion_points:
            row = ["—", "No discussion points captured.", "", "", "", "", "", "", "", "", "", "", ""]
            self._write_data_row(ws, 2, row, styles)
        else:
            for row_idx, dp in enumerate(discussion_points, start=2):
                num = row_idx - 1
                point = self._get(dp, "point", f"Discussion {num}")
                detailed_summary = self._get(dp, "detailed_summary", "")
                decision = self._get(dp, "decision", "No Decision Taken")
                task = self._get(dp, "task", "No Action Item")
                assigned_to = self._get(dp, "assigned_to", "Not Specified")
                deadline = self._get(dp, "deadline", "Not Specified")
                priority = self._get(dp, "priority", "Medium")
                status = self._get(dp, "status", "Open")
                risks_or_concerns = self._get(dp, "risks_or_concerns", "")
                suggestions = self._get(dp, "suggestions", "")
                follow_up_required = self._get(dp, "follow_up_required", "No")
                notes = self._get(dp, "notes", "")

                row = [
                    num,
                    point,
                    detailed_summary,
                    decision,
                    task,
                    assigned_to,
                    deadline,
                    priority,
                    status,
                    risks_or_concerns,
                    suggestions,
                    follow_up_required,
                    notes,
                ]
                self._write_data_row(ws, row_idx, row, styles, row_height=40)

        ws.freeze_panes = "C2"
        self._autofit(ws, max_width=60)

    def _build_pending_parking(self, ws, summary_data: dict, styles: dict) -> None:
        """Sheet 3: Pending Items and Parking Lot Topics."""
        ws.views.sheetView[0].showGridLines = True

        pending_items = summary_data.get("pending_items", [])
        parking_lot = summary_data.get("parking_lot", [])

        # ── Pending Items section ──
        header_font = styles["header_font"]
        header_fill = styles["header_fill"]
        center = styles["center_align"]
        left = styles["left_align"]
        border = styles["thin_border"]
        data_font = styles["data_font"]

        # Section heading row
        ws.append(["PENDING ITEMS"])
        cell = ws.cell(row=1, column=1)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border
        ws.row_dimensions[1].height = 22

        # Pending column headers
        ws.append(["#", "Pending Item / Deferred Decision"])
        for col in range(1, 3):
            c = ws.cell(row=2, column=col)
            c.font = styles["sub_header_font"]
            c.fill = styles["sub_header_fill"]
            c.alignment = center
            c.border = border
        ws.row_dimensions[2].height = 20

        if pending_items:
            for i, item in enumerate(pending_items, start=1):
                ws.append([i, item])
                for col in range(1, 3):
                    c = ws.cell(row=2 + i, column=col)
                    c.font = data_font
                    c.alignment = left
                    c.border = border
                ws.row_dimensions[2 + i].height = 20
        else:
            ws.append(["—", "No pending items identified."])
            for col in range(1, 3):
                c = ws.cell(row=3, column=col)
                c.font = data_font
                c.alignment = left
                c.border = border

        # Gap row
        gap_row = ws.max_row + 2
        ws.append([])
        ws.append([])

        # ── Parking Lot section ──
        park_start = gap_row
        ws.append(["PARKING LOT TOPICS"])
        cell = ws.cell(row=park_start, column=1)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border
        ws.row_dimensions[park_start].height = 22

        ws.append(["#", "Parking Lot Topic"])
        for col in range(1, 3):
            c = ws.cell(row=park_start + 1, column=col)
            c.font = styles["sub_header_font"]
            c.fill = styles["sub_header_fill"]
            c.alignment = center
            c.border = border
        ws.row_dimensions[park_start + 1].height = 20

        if parking_lot:
            for i, item in enumerate(parking_lot, start=1):
                ws.append([i, item])
                for col in range(1, 3):
                    c = ws.cell(row=park_start + 1 + i, column=col)
                    c.font = data_font
                    c.alignment = left
                    c.border = border
                ws.row_dimensions[park_start + 1 + i].height = 20
        else:
            ws.append(["—", "No parking lot topics identified."])
            for col in range(1, 3):
                c = ws.cell(row=park_start + 2, column=col)
                c.font = data_font
                c.alignment = left
                c.border = border

        self._autofit(ws)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_styles(theme_hex: str) -> dict:
        """Build and return a shared styles dictionary."""
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )
        return {
            "header_font": Font(name="Calibri", size=11, bold=True, color="FFFFFF"),
            "header_fill": PatternFill(start_color=theme_hex, end_color=theme_hex, fill_type="solid"),
            "sub_header_font": Font(name="Calibri", size=10, bold=True, color="FFFFFF"),
            "sub_header_fill": PatternFill(start_color="4A6FA5", end_color="4A6FA5", fill_type="solid"),
            "data_font": Font(name="Calibri", size=10, color="000000"),
            "center_align": Alignment(horizontal="center", vertical="center", wrap_text=True),
            "left_align": Alignment(horizontal="left", vertical="top", wrap_text=True),
            "thin_border": thin_border,
        }

    @staticmethod
    def _write_headers(ws, headers: list, styles: dict) -> None:
        """Write a header row with theme fill and bold white font."""
        ws.append(headers)
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = styles["header_font"]
            cell.fill = styles["header_fill"]
            cell.alignment = styles["center_align"]
            cell.border = styles["thin_border"]
        ws.row_dimensions[1].height = 28

    @staticmethod
    def _write_data_row(ws, row_idx: int, row: list, styles: dict, row_height: int = 22) -> None:
        """Write a single data row with standard styling."""
        ws.append(row)
        for col_num in range(1, len(row) + 1):
            cell = ws.cell(row=row_idx, column=col_num)
            cell.font = styles["data_font"]
            cell.border = styles["thin_border"]
            cell.alignment = styles["left_align"]
        ws.row_dimensions[row_idx].height = row_height

    @staticmethod
    def _autofit(ws, max_width: int = 50) -> None:
        """Auto-fit column widths based on content, capped at max_width."""
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max((len(str(cell.value or "")) for cell in col), default=0)
            ws.column_dimensions[col_letter].width = max(min(max_len + 3, max_width), 12)

    @staticmethod
    def _get(item, key: str, default: str = "") -> str:
        """Safely get a value from a dict or Pydantic model."""
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    @staticmethod
    def _is_task(owner: str, status: str) -> bool:
        """Return True if the row represents an assignable task (not an information row)."""
        if status and status.strip().lower() in ("information", "info"):
            return False
        if not owner or not owner.strip():
            return False
        return owner.strip().lower() not in (
            "unknown", "general", "none", "n/a", "nil",
            "general discussion", "information", "info", "",
        )
