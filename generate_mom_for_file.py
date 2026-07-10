import sys
from pathlib import Path
from ai.pipeline.manager import AIManager
from reports.report_manager import ReportManager

def main():
    transcript_path = Path(r"C:\Users\Vaps\PycharmProjects\AIMOM\output\AI_Review_Meeting_20260708_141745.txt")
    if not transcript_path.exists():
        print(f"Error: {transcript_path} does not exist.")
        sys.exit(1)
        
    print(f"Reading transcript from {transcript_path}...")
    transcript = transcript_path.read_text(encoding="utf-8")
    
    print("Initializing AIManager and ReportManager...")
    ai_manager = AIManager()
    report_manager = ReportManager()
    
    title = "AI Review Meeting"
    date = "2026-07-08"
    
    print(f"Analyzing meeting '{title}' for date '{date}' (respecting the 20-second API delay)...")
    try:
        summary = ai_manager.analyze_meeting(
            title=title,
            date=date,
            transcript=transcript,
        )
    except Exception as exc:
        print(f"Error during AI analysis: {exc}")
        sys.exit(1)
        
    print("Generating reports (PDF + Excel)...")
    try:
        reports = report_manager.generate_reports(summary)
    except Exception as exc:
        print(f"Error during report generation: {exc}")
        sys.exit(1)
        
    print("\nGeneration completed successfully!")
    print(f"PDF Report: {reports['pdf']}")
    print(f"Excel Tracker: {reports['excel']}")

if __name__ == "__main__":
    main()
