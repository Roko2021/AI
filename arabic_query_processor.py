import re
import pyarabic.araby as araby


# ==========================================
# Arabic Query Processor
# ==========================================

def normalize_arabic(text):
    """
    تنظيف وتوحيد النص العربي
    """

    # إزالة التشكيل
    text = araby.strip_tashkeel(text)

    # إزالة التطويل
    text = araby.strip_tatweel(text)

    # توحيد الهمزات
    text = araby.normalize_hamza(text)

    # توحيد أشكال الألف
    text = re.sub(r"[إأآٱ]", "ا", text)

    # توحيد الياء والألف المقصورة
    text = text.replace("ى", "ي")

    # إزالة علامات الترقيم الزائدة
    text = re.sub(
        r"[^\w\s\u0600-\u06FF؟?!.,]",
        " ",
        text
    )

    # إزالة المسافات المتكررة
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# Common Arabic Typing Corrections
# ==========================================

COMMON_CORRECTIONS = {

    # أخطاء شائعة
    "ابز": "ابرز",
    "ابرز": "ابرز",

    "استخدامت": "استخدامات",
    "استخدامات": "استخدامات",

    "استخدام": "استخدام",

    "ماهى": "ما هي",
    "ماهي": "ما هي",

    "ازاى": "ازاي",
    "ازاي": "ازاي",

    "ايه": "ايه",
    "اية": "ايه",

    "الذكاء الاصطناعي": "الذكاء الاصطناعي",
    "الذكاء الاصنطاعي": "الذكاء الاصطناعي",

    "بايثن": "بايثون",
    "بايثون": "بايثون",

    "برمجه": "برمجة",
    "برمجة": "برمجة",

    "البرمجه": "البرمجة",
    "البرمجة": "البرمجة",

    "معلوملت": "معلومات",
    "معلومات": "معلومات",

    "تطبيقات": "تطبيقات",
    "تطبيقلت": "تطبيقات",

    "موقع": "موقع",
    "مواقع": "مواقع",

    "بيانات": "بيانات",
    "بيانت": "بيانات",

    "تحليل": "تحليل",
    "تحلييل": "تحليل",
}


# ==========================================
# Correct Individual Words
# ==========================================

def correct_words(text):

    words = text.split()

    corrected_words = []

    for word in words:

        # نحافظ على علامات الاستفهام
        punctuation = ""

        if word.endswith("؟"):
            punctuation = "؟"
            word = word[:-1]

        elif word.endswith("?"):
            punctuation = "?"
            word = word[:-1]

        corrected = COMMON_CORRECTIONS.get(
            word,
            word
        )

        corrected_words.append(
            corrected + punctuation
        )

    return " ".join(corrected_words)


# ==========================================
# Full Processing
# ==========================================

def process_query(text):

    original = text

    # Step 1
    normalized = normalize_arabic(text)

    # Step 2
    corrected = correct_words(normalized)

    # تنظيف نهائي
    corrected = re.sub(
        r"\s+",
        " ",
        corrected
    ).strip()

    return {
        "original": original,
        "normalized": normalized,
        "corrected": corrected
    }


# ==========================================
# Interactive Test
# ==========================================

if __name__ == "__main__":

    print("=" * 70)
    print("Arabic Query Processor")
    print("=" * 70)

    print("\nEnter Arabic questions.")
    print("Type exit to stop.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = process_query(question)

        print("\nOriginal:")
        print(result["original"])

        print("\nNormalized:")
        print(result["normalized"])

        print("\nCorrected:")
        print(result["corrected"])

        print("\n" + "-" * 70)