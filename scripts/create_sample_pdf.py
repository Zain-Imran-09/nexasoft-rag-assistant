
from pathlib import Path

from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent

fixture_dir = PROJECT_ROOT / "tests" / "fixtures"
fixture_dir.mkdir(parents=True, exist_ok=True)

pdf_path = fixture_dir / "sample_company.pdf"

pdf = canvas.Canvas(str(pdf_path))

# Add text to the first PDF page.
pdf.drawString(72, 750, "NexaSoft - Test Company Profile")
pdf.drawString(
    72, 720,
    "NexaSoft provides AI chatbot development."
)

pdf.showPage()

# Add text to the second PDF page.
pdf.drawString(72, 750, "NexaSoft - Contact")
pdf.drawString(
    72, 720,
    "NexaSoft is based in Lahore, Pakistan."
)

pdf.save()

print("Created:", pdf_path)
