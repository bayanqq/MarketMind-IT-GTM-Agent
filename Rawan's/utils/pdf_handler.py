"""
utils/pdf_handler.py
--------------------
Safe PDF text extraction layer for MarketMind AI.

Wraps PyPDF with graceful error handling so that upstream agents receive
clean, validated text without having to manage IO or parsing exceptions.
"""

from pathlib import Path

from pypdf import PdfReader

from utils.security import validate_input_text, MAX_INPUT_LENGTH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract and return all text from a PDF file.

    Iterates over every page, concatenates the extracted text, and returns
    it after truncating to :data:`~utils.security.MAX_INPUT_LENGTH` characters
    to stay within downstream API limits.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing the full document text (up to the safety
        limit).

    Raises:
        FileNotFoundError: When *file_path* does not exist.
        ValueError: When the PDF contains no extractable text (e.g. scanned
            image-only PDF without OCR).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found at path: {path.resolve()}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: '{path.suffix}'")

    reader = PdfReader(str(path))

    pages_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[PDF Handler] Warning: could not extract text from page "
                f"{page_number} — {exc}"
            )

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in the PDF. "
            "The file may be a scanned image without embedded text."
        )

    # Truncate to the security maximum so downstream validation always passes.
    if len(full_text) > MAX_INPUT_LENGTH:
        print(
            f"[PDF Handler] Extracted text ({len(full_text)} chars) truncated "
            f"to {MAX_INPUT_LENGTH} chars."
        )
        full_text = full_text[:MAX_INPUT_LENGTH]

    return full_text


def load_and_validate_pdf(file_path: str | Path) -> str | None:
    """Extract PDF text and run it through the security validator.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The validated text string, or ``None`` when extraction fails or the
        text fails security validation.
    """
    try:
        text = extract_text_from_pdf(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[PDF Handler] Extraction failed: {exc}")
        return None

    if not validate_input_text(text):
        print("[PDF Handler] Extracted PDF text failed security validation.")
        return None

    return text
