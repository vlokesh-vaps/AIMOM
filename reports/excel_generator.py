"""Excel Action Tracker Generator using openpyxl.

Outputs structured action items into a formatted Excel action tracker spreadsheet.
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
    """Excel generator creating stylized corporate action trackers from meeting action items."""

    def __init__(self) -> None:
        self._output_dir = Path(__file__).resolve().parent / "excel"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, summary_data: dict, filename: str) -> Path:
        """Generate a styled corporate Excel action tracker spreadsheet.

        Args:
            summary_data: Dictionary conforming to MeetingSummary schema.
            filename: Output filename (e.g. Review_Meeting_2026-06-30.xlsx).

        Returns:
            Path to the saved Excel workbook.
        """
        start_time = time.time()
        output_path = self._output_dir / filename

        wb = Workbook()
        ws = wb.active
        ws.title = "Action Tracker"

        # Enable grid lines visibility
        ws.views.sheetView[0].showGridLines = True

        # Header Columns
        headers = [
            "Task(Review Meeting)",
            "Empcode",
            "Type",
            "Assigned To",
            "ActionPlans",
            "TargetDate",
        ]

        # Style Definitions
        theme_hex = COMPANY_THEME_COLOR.lstrip("#")
        # Ensure we have a valid hex color for Excel
        if len(theme_hex) != 6:
            theme_hex = "1E3A8A"  # Default dark blue fallback

        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color=theme_hex, end_color=theme_hex, fill_type="solid")
        
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="top", wrap_text=True)
        
        thin_border = Border(
            left=Side(style='thin', color='D9D9D9'),
            right=Side(style='thin', color='D9D9D9'),
            top=Side(style='thin', color='D9D9D9'),
            bottom=Side(style='thin', color='D9D9D9')
        )

        # 1. Write headers
        ws.append(headers)
        
        # Style headers
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border
        
        ws.row_dimensions[1].height = 28

        # 2. Write data rows
        action_items = summary_data.get("action_items", [])
        meeting_title = summary_data.get("meeting_title", "Review Meeting")
        participants = summary_data.get("participants", [])
        default_participant = participants[0] if participants else "Unknown"

        data_row_font = Font(name="Calibri", size=11, color="000000")

        if not action_items:
            # If no action items, write a single placeholder row
            ws.append([meeting_title, "", "Information", "", "No action items identified.", ""])
            for col_num in range(1, len(headers) + 1):
                cell = ws.cell(row=2, column=col_num)
                cell.font = data_row_font
                cell.alignment = left_align
                cell.border = thin_border
            ws.row_dimensions[2].height = 20
        else:
            row_idx = 2
            for item in action_items:
                # Handle dict vs pydantic model
                task_desc = item.get("task", "") if isinstance(item, dict) else getattr(item, "task", "")
                owner = item.get("owner", "") if isinstance(item, dict) else getattr(item, "owner", "")
                target_date = item.get("target_date", "") if isinstance(item, dict) else getattr(item, "target_date", "")
                status = item.get("status", "") if isinstance(item, dict) else getattr(item, "status", "")

                # Determine if it's a task or general discussion
                is_task = False
                if owner and owner.strip():
                    o_lower = owner.strip().lower()
                    if o_lower not in ("unknown", "general", "none", "n/a", "nil", "general discussion", "information", "info"):
                        is_task = True
                if status and status.strip().lower() in ("information", "info"):
                    is_task = False

                if is_task:
                    participant_value = owner.strip()
                    row_type = "Task"
                else:
                    participant_value = ""
                    row_type = "Information"

                row_data = [
                    meeting_title,        # Task(Review Meeting) column
                    "",                  # Empcode (manual entry)
                    row_type,            # Type (Task vs General Discussion)
                    participant_value,   # Assigned To (only the task receiver/assignee)
                    task_desc,           # ActionPlans
                    target_date if is_task else "",  # TargetDate
                ]

                ws.append(row_data)

                # Format data row cells
                for col_num in range(1, len(headers) + 1):
                    cell = ws.cell(row=row_idx, column=col_num)
                    cell.font = data_row_font
                    cell.border = thin_border
                    
                    # Align target date, type, and empty cells to center/left accordingly
                    if col_num in (2, 3, 6):
                        cell.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
                    else:
                        cell.alignment = left_align

                ws.row_dimensions[row_idx].height = 22
                row_idx += 1

        # 3. Apply sheet filters
        last_row = len(action_items) + 1 if action_items else 2
        ws.auto_filter.ref = f"A1:F{last_row}"

        # 4. Freeze top header row
        ws.freeze_panes = "A2"

        # 5. Autofit column widths dynamically
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Avoid errors with empty/None cells
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            # Add padding, enforce min and max column widths
            ws.column_dimensions[col_letter].width = max(min(max_len + 3, 50), 12)

        # Save workbook
        wb.save(str(output_path))

        # Metrics logging
        duration = time.time() - start_time
        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(
            "Excel generation complete. Path=%s, Size=%.2f KB, Duration=%.2fs",
            output_path,
            file_size,
            duration,
        )
        return output_path
