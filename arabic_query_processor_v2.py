import re


# ============================================================
# 1. Normalization
# ============================================================

def normalize_arabic(text):
    """
    تنظيف وتوحيد بعض الاختلافات الشائعة في الكتابة العربية.
    """

    # إزالة التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # توحيد بعض الحروف
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # الياء والألف المقصورة
    text = text.replace("ى", "ي")

    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# 2. كلمات وتصحيحات مؤكدة
# ============================================================

CORRECTIONS = {

    # أخطاء بايثون
    "بايثن": "بايثون",
    "بايثونن": "بايثون",

    # استخدامات
    "استخدامت": "استخدامات",
    "استخدمات": "استخدامات",
    "استخدمات": "استخدامات",

    # أبرز
    "ابز": "ابرز",
    "ابرز": "ابرز",

    # البرمجة
    "البرمجه": "البرمجة",

    # البيانات
    "البيانت": "البيانات",

    # الذكاء الاصطناعي
    "الاصنطاعي": "الاصطناعي",

    # كلمات شائعة
    "مستخدمه": "مستخدمة",
    "مميزات": "مميزات",
}


# ============================================================
# 3. تصحيح النص
# ============================================================

def correct_text(text):
    """
    تصحيح الكلمات الموجودة في قاموس التصحيحات فقط.
    """

    words = text.split()

    corrected_words = []

    for word in words:

        # الاحتفاظ بعلامات الترقيم
        prefix = ""
        suffix = ""

        # علامات في بداية الكلمة
        while word and word[0] in "،؟!.,:؛()[]{}\"'":
            prefix += word[0]
            word = word[1:]

        # علامات في نهاية الكلمة
        while word and word[-1] in "،؟!.,:؛()[]{}\"'":
            suffix = word[-1] + suffix
            word = word[:-1]

        corrected_word = CORRECTIONS.get(word, word)

        corrected_words.append(
            prefix + corrected_word + suffix
        )

    return " ".join(corrected_words)


# ============================================================
# 4. Process Query
# ============================================================

def process_query(question):

    normalized = normalize_arabic(question)

    corrected = correct_text(normalized)

    return normalized, corrected


# ============================================================
# 5. Test
# ============================================================

def main():

    print("=" * 70)
    print("Arabic Query Processor V2")
    print("=" * 70)

    print("\nEnter Arabic questions.")
    print("Type exit to stop.\n")

    while True:

        question = input("Question: ")

        if question.lower() == "exit":
            break

        if not question.strip():
            continue

        normalized, corrected = process_query(question)

        print("\nOriginal:")
        print(question)

        print("\nNormalized:")
        print(normalized)

        print("\nCorrected:")
        print(corrected)

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()