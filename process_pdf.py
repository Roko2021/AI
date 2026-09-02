from pathlib import Path
import re

import pymupdf
import pytesseract
from pdf2image import convert_from_path


# ============================================================
# SETTINGS
# ============================================================

PDF_PATH = Path("documents/my_document.pdf")
OUTPUT_PATH = Path("cleaned_text.txt")

DPI = 300

# لو Tesseract مش موجود في PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# TEXT QUALITY
# ============================================================

def text_quality(text):
    """
    تقييم بسيط لجودة النص المستخرج.
    كلما زادت نسبة الحروف العربية والنص المفيد
    اعتبرنا النص أفضل.
    """

    if not text:
        return 0

    arabic_chars = len(
        re.findall(r"[\u0600-\u06FF]", text)
    )

    total_chars = len(text.strip())

    if total_chars == 0:
        return 0

    arabic_ratio = arabic_chars / total_chars

    return arabic_ratio


# ============================================================
# EXTRACT TEXT USING PYMUPDF
# ============================================================

def extract_with_pymupdf(pdf_path):

    print("\n[1] Extracting text using PyMuPDF...")

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        print(
            f"  Page {page_number}/{len(doc)}..."
        )

        text = page.get_text("text")

        text = text.strip()

        if text:
            pages.append(
                f"--- PAGE {page_number} ---\n{text}"
            )

    doc.close()

    result = "\n\n".join(pages)

    print(
        f"  Characters extracted: {len(result)}"
    )

    print(
        f"  Arabic ratio: {text_quality(result):.2%}"
    )

    return result


# ============================================================
# OCR USING TESSERACT
# ============================================================

def extract_with_ocr(pdf_path):

    print("\n[2] PyMuPDF quality is low.")
    print("    Starting Arabic OCR...")
    print(f"    DPI: {DPI}")

    print("\nConverting PDF pages to images...")

    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        fmt="png"
    )

    print(
        f"Converted pages: {len(pages)}"
    )

    all_pages = []

    for page_number, image in enumerate(
        pages,
        start=1
    ):

        print(
            f"\nOCR page "
            f"{page_number}/{len(pages)}..."
        )

        text = pytesseract.image_to_string(
            image,
            lang="ara+eng",
            config="--psm 6"
        )

        text = text.strip()

        print(
            f"Characters extracted: {len(text)}"
        )

        if text:

            all_pages.append(
                f"--- PAGE {page_number} ---\n{text}"
            )

    return "\n\n".join(all_pages)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    print("\n[3] Cleaning text...")

    # إزالة المسافات الزائدة
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # تقليل الأسطر الفارغة
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # إزالة المسافات قبل علامات الترقيم
    text = re.sub(
        r"\s+([،؛:!?؟])",
        r"\1",
        text
    )

    # إصلاح بعض المسافات الشائعة داخل الكلمات
    replacements = {
        "حس وب": "حسوب",
        "توف ير": "توفير",
        "عالي ة": "عالية",
        "مج االت": "مجالات",
        "البرمج ة": "البرمجة",
        "التط بيق": "التطبيق",
        "األكاديمية": "الأكاديمية",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SMART PDF TEXT EXTRACTION")
    print("=" * 60)

    print(f"\nPDF: {PDF_PATH}")

    if not PDF_PATH.exists():

        print(
            f"\nERROR: PDF not found:"
            f"\n{PDF_PATH}"
        )

        return

    # --------------------------------------------------------
    # First attempt: PyMuPDF
    # --------------------------------------------------------

    text = extract_with_pymupdf(
        PDF_PATH
    )

    quality = text_quality(text)

    # --------------------------------------------------------
    # Decide whether OCR is needed
    # --------------------------------------------------------

    MIN_CHARACTERS = 1000
    MIN_ARABIC_RATIO = 0.20

    use_ocr = (
        len(text) < MIN_CHARACTERS
        or quality < MIN_ARABIC_RATIO
    )

    if use_ocr:

        text = extract_with_ocr(
            PDF_PATH
        )

    else:

        print(
            "\n[OK] PyMuPDF extraction "
            "is good enough."
        )

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    text = clean_text(text)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"Total characters: {len(text)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)
    print("TEXT PREVIEW")
    print("=" * 60)

    print(
        text[:5000]
    )


if __name__ == "__main__":
    main()