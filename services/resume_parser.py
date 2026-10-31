import io
import os
import re
import shutil
import subprocess
from pathlib import Path

# Safe pymupdf import without deprecation warning
try:
    import pymupdf as fitz
except ImportError:
    import fitz

import docx
import pytesseract
from PIL import Image

# Locate tesseract OCR safely
default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(default_tesseract):
    pytesseract.pytesseract.tesseract_cmd = default_tesseract
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


def extract_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)

    for page in pdf:
        page_text = page.get_text().strip()

        if page_text:
            text += page_text + "\n"
        else:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    text += ocr_text + "\n"
            except Exception:
                # OCR is optional fallback; don't crash if page cannot be OCR'd
                pass

    pdf.close()
    return text


def extract_docx(file_path):
    document = docx.Document(file_path)
    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                text.append(" | ".join(row_cells))

    return "\n".join(text)


def extract_doc(file_path):
    output_folder = str(Path(file_path).parent)

    soffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_cmd:
        raise ValueError("DOC format requires LibreOffice/soffice. Please convert file to .docx or .pdf.")

    result = subprocess.run(
        [
            soffice_cmd,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            output_folder,
            file_path
        ],
        capture_output=True,
        text=True
    )

    converted_file = Path(file_path).with_suffix(".docx")

    if result.returncode != 0 or not converted_file.exists():
        raise ValueError("Could not convert DOC file. Install LibreOffice first.")

    try:
        text = extract_docx(str(converted_file))
    finally:
        if converted_file.exists():
            try:
                converted_file.unlink()
            except Exception:
                pass

    return text


def extract_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)


def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def clean_resume_text(text):
    """
    Cleans resume text by removing non-printable/corrupted artifacts
    and normalizing whitespace, while preserving all valuable candidate information
    (names, titles, locations, contact info, skills, achievements).
    """
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove font icon artifacts mapped to single characters before contacts
        line = re.sub(r"^[^\w\s\(\)\[\]\{\}\<\>@\+\-\*\•–—]+", "", line).strip()
        line = re.sub(r"^[A-Za-zÓ®•·§]\s+(?=[+\d]|https?://|[a-zA-Z0-9._%+-]+@)", "", line).strip()

        # Normalize multiple spaces and tabs to single spaces
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line and len(line) >= 2:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_resume(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        text = extract_pdf(file_path)
    elif extension == ".docx":
        text = extract_docx(file_path)
    elif extension == ".doc":
        text = extract_doc(file_path)
    elif extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        text = extract_image(file_path)
    elif extension == ".txt":
        text = extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    text = clean_resume_text(text)

    if not text.strip():
        raise ValueError("No text could be extracted from this resume")

    return text
