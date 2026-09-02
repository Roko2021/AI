from difflib import get_close_matches


# كلمات عربية صحيحة نريد استخدامها كمرجع
ARABIC_DICTIONARY = [
    "الذكاء",
    "الاصطناعي",
    "استخدامات",
    "بايثون",
    "البرمجة",
    "البيانات",
    "تحليل",
    "تطبيقات",
    "لغات",
    "مميزات",
    "تعلم",
    "تطوير",
    "المواقع",
    "الويب",
    "الآلة",
    "تعلم",
    "البرامج",
]


def correct_word(word):
    """
    محاولة تصحيح كلمة عربية إذا كانت قريبة جدًا
    من كلمة موجودة في القاموس.
    """

    clean_word = word.strip("،؟!.,:؛()[]{}\"'")

    matches = get_close_matches(
        clean_word,
        ARABIC_DICTIONARY,
        n=1,
        cutoff=0.75
    )

    if matches:
        return matches[0]

    return clean_word


def correct_text(text):
    words = text.split()

    corrected_words = []

    for word in words:
        corrected_words.append(correct_word(word))

    return " ".join(corrected_words)


def main():

    print("=" * 70)
    print("Arabic Spell Correction Test")
    print("=" * 70)

    questions = [
        "ما هو الذكاء الاصنطاعي؟",
        "ما هى ابرز استخدمات بايثون؟",
        "ما هى ابرز استخدامات بايثون؟",
        "ما هى مميزات لغة بايثن؟",
        "كيف اتعلم البرمجه بلغة بايثن؟",
        "ما هى تطبيقات الذكاء الاصنطاعي؟",
        "ما هى لغات البرمجه المستخدمه فى الذكاء الاصنطاعي؟",
        "ما هو سعر الدولار اليوم؟",
    ]

    for question in questions:

        corrected = correct_text(question)

        print("\nOriginal:")
        print(question)

        print("\nCorrected:")
        print(corrected)

        print("-" * 70)


if __name__ == "__main__":
    main()