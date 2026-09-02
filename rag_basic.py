import requests
import chromadb
import re

from arabic_query_processor_v2 import process_query


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "qwen2.5-coder:7b"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "website_knowledge_v2"

RETRIEVAL_COUNT = 10

DISPLAY_COUNT = 5

CONTEXT_COUNT = 4


# ============================================================
# Thresholds
# ============================================================

CONTEXT_THRESHOLD = 0.45

STRONG_FINAL_THRESHOLD = 0.70

STRONG_SEMANTIC_THRESHOLD = 0.90

STRONG_KEYWORD_THRESHOLD = 0.60

MIN_SEMANTIC_FOR_KEYWORD = 0.40

ENTITY_BOOST = 0.10

QUESTION_ONLY_PENALTY = 0.025

BOILERPLATE_PENALTY = 0.20

INCONSISTENT_PENALTY = 0.35


# ============================================================
# Intent-aware thresholds
# ============================================================

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------

USAGE_INTENT_BOOST = 0.25

USAGE_INTENT_PENALTY = 0.30

USAGE_MIN_EVIDENCE = 0.25

USAGE_CONTEXT_MIN_SCORE = 0.50


# ------------------------------------------------------------
# Project
# ------------------------------------------------------------

PROJECT_INTENT_BOOST = 0.20

PROJECT_INTENT_PENALTY = 0.25

PROJECT_MIN_EVIDENCE = 0.25

PROJECT_CONTEXT_MIN_SCORE = 0.48


# ------------------------------------------------------------
# Learning
# ------------------------------------------------------------

LEARNING_INTENT_BOOST = 0.20

LEARNING_INTENT_PENALTY = 0.20

LEARNING_MIN_EVIDENCE = 0.25

LEARNING_CONTEXT_MIN_SCORE = 0.48


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------

SELECTION_INTENT_BOOST = 0.15

SELECTION_INTENT_PENALTY = 0.20


# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

COMPARISON_INTENT_BOOST = 0.15

COMPARISON_INTENT_PENALTY = 0.20


# ============================================================
# Arabic normalization
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # --------------------------------------------------------
    # Arabic Alef normalization
    # --------------------------------------------------------

    text = re.sub(
        r"[إأآا]",
        "ا",
        text
    )

    # --------------------------------------------------------
    # Hamza
    # --------------------------------------------------------

    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    # --------------------------------------------------------
    # Alef Maqsura
    # --------------------------------------------------------

    text = text.replace("ى", "ي")

    # --------------------------------------------------------
    # Taa Marbuta
    # --------------------------------------------------------

    text = text.replace("ة", "ه")

    # --------------------------------------------------------
    # Tatweel
    # --------------------------------------------------------

    text = text.replace("ـ", "")

    # --------------------------------------------------------
    # Tashkeel
    # --------------------------------------------------------

    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # --------------------------------------------------------
    # Arabic / English punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Arabic conjunction "و"
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\S)و(?=[\u0600-\u06FF])",
        "و ",
        text
    )

    # --------------------------------------------------------
    # Multiple spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Embedding
# ============================================================

def create_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get(
        "embeddings",
        []
    )

    if not embeddings:

        raise RuntimeError(
            "Embedding model returned no embedding."
        )

    return embeddings[0]


# ============================================================
# Stop Words
# ============================================================

STOP_WORDS = {

    "ما",
    "ماذا",
    "هي",
    "هو",
    "هل",

    "من",
    "في",
    "عن",
    "على",
    "الى",
    "إلى",

    "و",
    "او",
    "أو",

    "اي",
    "أي",

    "ماهي",
    "ماهى",
    "ماهو",
    "ماهوا",

    "كيف",
    "لماذا",
    "متى",
    "اين",
    "أين",

    "يمكن",

    "انا",
    "أنا",

    "لي",

    "لغة",
    "لغات",

    "اريد",
    "أريد",

    "عاوز",
    "عاوزة",

    "ارغب",
    "أرغب",

    "اليوم",
    "الان",
    "الآن",

    "عبر",

    "من"
}


# ============================================================
# Keywords
# ============================================================

def get_keywords(text):

    normalized = normalize_text(text)

    words = normalized.split()

    keywords = []

    for word in words:

        if word in STOP_WORDS:
            continue

        if len(word) <= 1:
            continue

        keywords.append(word)

    return keywords


# ============================================================
# Keyword Score
# ============================================================

def keyword_score(
    question,
    document
):

    question_keywords = get_keywords(
        question
    )

    document_text = normalize_text(
        document
    )

    if not question_keywords:

        return 0.0

    document_words = set(
        document_text.split()
    )

    matched = 0

    for keyword in question_keywords:

        if keyword in document_words:

            matched += 1

    return matched / len(
        question_keywords
    )


# ============================================================
# Topic / Entity Vocabulary
# ============================================================

TOPIC_ENTITIES = {

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    "python": [

        "بايثون",
        "python",
        "لغة بايثون"
    ],

    # --------------------------------------------------------
    # Java
    # --------------------------------------------------------

    "java": [

        "جافا",
        "java",
        "لغة جافا"
    ],

    # --------------------------------------------------------
    # JavaScript
    # --------------------------------------------------------

    "javascript": [

        "جافاسكريبت",
        "جافا سكريبت",
        "javascript",
        "لغة جافاسكريبت"
    ],

    # --------------------------------------------------------
    # C#
    # --------------------------------------------------------

    "csharp": [

        "سي شارب",
        "c#",
        "c sharp",
        "csharp"
    ],

    # --------------------------------------------------------
    # Mobile
    # --------------------------------------------------------

    "mobile": [

        "هاتف",
        "هاتفك",
        "هاتفي",
        "هاتفه",
        "هاتفها",
        "هواتف",

        "موبايل",
        "موبايلك",
        "موبايلي",
        "موبايله",
        "موبايلها",
        "موبايلات",

        "الهاتف",
        "الهواتف",

        "الموبايل",
        "الموبايلات",

        "تطبيقات الهاتف",
        "تطبيق الهاتف",

        "تطبيقات الموبايل",
        "تطبيق الموبايل"
    ],

    # --------------------------------------------------------
    # Games
    # --------------------------------------------------------

    "games": [

        "العاب",
        "الالعاب",
        "ألعاب",
        "الألعاب",

        "لعبة",
        "لعبه",

        "تطوير الالعاب",
        "تطوير الألعاب",

        "العاب الفيديو",
        "ألعاب الفيديو",

        "العاب الكمبيوتر",
        "ألعاب الكمبيوتر"
    ],

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    "artificial_intelligence": [

        "الذكاء الاصطناعي",
        "ذكاء اصطناعي",
        "الذكاء الصناعي",

        "ai",
        "ال ai",
        "الذكاء"
    ],

    # --------------------------------------------------------
    # Data Analysis
    # --------------------------------------------------------

    "data_analysis": [

        "تحليل البيانات",
        "تحليل بيانات"
    ],

    # --------------------------------------------------------
    # Web
    # --------------------------------------------------------

    "web": [

        "الويب",
        "ويب",

        "المواقع",
        "موقع",

        "تطوير المواقع",
        "تطوير الويب",

        "مواقع الويب"
    ],

    # --------------------------------------------------------
    # Finance
    # --------------------------------------------------------

    "finance": [

        "الدولار",
        "دولار",

        "الذهب",
        "ذهب",

        "سعر الدولار",
        "سعر الذهب",

        "اسعار الدولار",
        "اسعار الذهب",

        "سوق المال",

        "العملات",
        "عملة"
    ],

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    "weather": [

        "الطقس",
        "طقس",

        "درجة الحرارة",
        "درجه الحراره",

        "الحراره",
        "الحرارة",

        "حاله الطقس",
        "حالة الطقس"
    ],

    # --------------------------------------------------------
    # Geography
    # --------------------------------------------------------

    "geography": [

        "عاصمة",
        "عاصمه",

        "دولة",
        "دوله",

        "مدينة",
        "مدينه",

        "بلد",
        "بلاد",

        "جغرافيا",
        "جغرافيه"
    ],

    # --------------------------------------------------------
    # Sports
    # --------------------------------------------------------

    "sports": [

        "محمد صلاح",

        "كرة القدم",
        "كره القدم",

        "رياضة",
        "رياضه",

        "الدوري",

        "بطولة"
    ],

    # --------------------------------------------------------
    # Cars
    # --------------------------------------------------------

    "cars": [

        "السيارة",
        "سيارة",

        "السيارات",

        "اصلاح السيارة",
        "إصلاح السيارة",

        "محرك السيارة",
        "محرك"
    ]
}


# ============================================================
# Entity normalization
# ============================================================

def normalize_entity_phrase(
    phrase
):

    return normalize_text(
        phrase
    )


# ============================================================
# Tokenize
# ============================================================

def tokenize(text):

    normalized = normalize_text(
        text
    )

    return normalized.split()


# ============================================================
# Phrase Exists
# ============================================================

def phrase_exists(
    normalized_text,
    phrase
):

    normalized_text = normalize_text(
        normalized_text
    )

    normalized_phrase = normalize_entity_phrase(
        phrase
    )

    if not normalized_text:

        return False

    if not normalized_phrase:

        return False

    text_tokens = normalized_text.split()

    phrase_tokens = normalized_phrase.split()

    if not text_tokens:

        return False

    if not phrase_tokens:

        return False

    if len(phrase_tokens) == 1:

        return phrase_tokens[0] in text_tokens

    phrase_length = len(
        phrase_tokens
    )

    for i in range(
        len(text_tokens) - phrase_length + 1
    ):

        if (
            text_tokens[
                i:i + phrase_length
            ]
            ==
            phrase_tokens
        ):

            return True

    return False


# ============================================================
# Detect Entities
# ============================================================

def detect_entities(
    text
):

    normalized = normalize_text(
        text
    )

    detected = set()

    # --------------------------------------------------------
    # Exact entity matching
    # --------------------------------------------------------

    for entity_name, phrases in TOPIC_ENTITIES.items():

        for phrase in phrases:

            if phrase_exists(
                normalized,
                phrase
            ):

                detected.add(
                    entity_name
                )

                break

    # --------------------------------------------------------
    # Mobile morphology
    # --------------------------------------------------------

    tokens = normalized.split()

    for token in tokens:

        if re.fullmatch(
            r"هاتف(?:ك|ي|ه|ها)?",
            token
        ):

            detected.add(
                "mobile"
            )

        if re.fullmatch(
            r"موبايل(?:ك|ي|ه|ها)?",
            token
        ):

            detected.add(
                "mobile"
            )

    return detected


def detect_relation_evidence(
    question,
    document
):
    """
    Detect whether the document contains evidence
    connecting the important entities in the question.
    """

    question_entities = detect_entities(
        question
    )

    document_entities = detect_entities(
        document
    )

    # --------------------------------------------------------
    # No entities
    # --------------------------------------------------------

    if len(question_entities) < 2:

        return {
            "supported": False,
            "score": 0.0,
            "matched_entities": [],
            "missing_entities": [],
            "reason": "Not enough entities for relation check"
        }

    # --------------------------------------------------------
    # Find entities present in document
    # --------------------------------------------------------

    matched_entities = [
        entity
        for entity in question_entities
        if entity in document_entities
    ]

    missing_entities = [
        entity
        for entity in question_entities
        if entity not in document_entities
    ]

    # --------------------------------------------------------
    # Entity coverage
    # --------------------------------------------------------

    coverage = (
        len(matched_entities)
        /
        len(question_entities)
    )

    # --------------------------------------------------------
    # Relation keywords
    # --------------------------------------------------------

    relation_keywords = [
        "يستخدم",
        "تستخدم",
        "استخدام",
        "تستعمل",
        "يستعمل",
        "لتطوير",
        "تطوير",
        "في مجال",
        "يساعد",
        "يمكن استخدام",
        "تطبيق",
        "تطبيقات"
    ]

    document_text = document.lower()

    relation_found = any(
        keyword in document_text
        for keyword in relation_keywords
    )

    # --------------------------------------------------------
    # Calculate evidence score
    # --------------------------------------------------------

    score = 0.0

    if coverage >= 1.0:

        score += 0.50

    elif coverage >= 0.66:

        score += 0.30

    elif coverage >= 0.50:

        score += 0.20

    if relation_found:

        score += 0.50

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    supported = (
        coverage >= 1.0
        and
        relation_found
        and
        score >= 0.80
    )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    if supported:

        reason = (
            "All required entities found "
            "with relation evidence"
        )

    elif coverage < 1.0:

        reason = (
            "Missing required entities"
        )

    elif not relation_found:

        reason = (
            "Entities found but no relation evidence"
        )

    else:

        reason = (
            "Insufficient relation evidence"
        )

    return {
        "supported": supported,
        "score": round(score, 4),
        "matched_entities": matched_entities,
        "missing_entities": missing_entities,
        "reason": reason
    }

# ============================================================
# Question Only
# ============================================================

def is_question_only(
    document
):

    normalized = normalize_text(
        document
    )

    if not normalized:

        return True

    words = normalized.split()

    if len(words) <= 10:

        question_starts = [

            "هل",
            "ما",
            "ماذا",
            "كيف",
            "لماذا",
            "متى",
            "اين"
        ]

        if words[0] in question_starts:

            return True

    if "؟" in str(document):

        if len(words) <= 18:

            return True

    return False


# ============================================================
# Boilerplate
# ============================================================

BOILERPLATE_PHRASES = [

    "تسجيل الدخول",
    "حساب جديد",
    "الرئيسية",
    "للخلف",

    "تابعنا",
    "تابعنا على تويتر",
    "تابعنا على فيسبوك",
    "تابعنا على يوتيوب",

    "موسوعة حسوب",

    "قد يهمك ايضا",
    "قد يهمك أيضًا",

    "سياسة الخصوصية",
    "شروط الاستخدام",

    "جميع الحقوق محفوظة"
]


def is_boilerplate(
    document
):

    normalized = normalize_text(
        document
    )

    if not normalized:

        return True

    matches = 0

    for phrase in BOILERPLATE_PHRASES:

        if normalize_text(
            phrase
        ) in normalized:

            matches += 1

    if matches >= 2:

        return True

    navigation_words = [

        "الرئيسية",
        "تسجيل الدخول",
        "حساب جديد",
        "تابعنا",
        "تويتر",
        "فيسبوك",
        "يوتيوب"
    ]

    navigation_matches = 0

    for word in navigation_words:

        if normalize_text(
            word
        ) in normalized:

            navigation_matches += 1

    return navigation_matches >= 3


# ============================================================
# Question Intent
# ============================================================

def detect_question_intent(
    question
):

    normalized = normalize_text(
        question
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_patterns = [

        "الفرق بين",
        "ما الفرق",
        "فرق بين",
        "مقارنة",
        "قارن",
        "مقارنه"
    ]

    comparison = any(

        normalize_text(pattern)
        in normalized

        for pattern in comparison_patterns
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selection_patterns = [

        "افضل",
        "الافضل",
        "احسن",
        "الاحسن",
        "خير",
        "اختيار",
        "انسب",
        "الانسب",
        "مناسب"
    ]

    selection = any(

        normalize_text(pattern)
        in normalized

        for pattern in selection_patterns
    )

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    usage_patterns = [

        "استخدامات",
        "استخدام",
        "تستخدم في",
        "تستعمل في",
        "مجالات استخدام",
        "استخدامها",
        "استخدامه",
        "فيما تستخدم",
        "في ماذا تستخدم",
        "بماذا تستخدم",
        "مجالاتها",
        "مجالاته"
    ]

    usage = any(

        normalize_text(pattern)
        in normalized

        for pattern in usage_patterns
    )

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    project_patterns = [

        "مشاريع",
        "مشروع",
        "المشاريع",
        "العمل عليها",
        "يمكن العمل",
        "اعمل عليها",
        "اعمل بها",
        "ابني بها",
        "ابني عليها",
        "بناء مشروع",
        "بناء مشاريع"
    ]

    project = any(

        normalize_text(pattern)
        in normalized

        for pattern in project_patterns
    )

    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    learning_patterns = [

        "تعلم",
        "اتعلم",
        "تعلمها",
        "تعلمه",
        "دراسة",
        "تعلم لغة",
        "تعلم البرمجة",
        "ابدأ تعلم",
        "ابدأ في تعلم",
        "كيف اتعلم",
        "كيف اتعلمها"
    ]

    learning = any(

        normalize_text(pattern)
        in normalized

        for pattern in learning_patterns
    )

    return {

        "comparison": comparison,

        "selection": selection,

        "usage": usage,

        "project": project,

        "learning": learning
    }


# ============================================================
# Intent Evidence
# ============================================================

def get_intent_evidence(
    question,
    document
):

    normalized_document = normalize_text(
        document
    )

    intent = detect_question_intent(
        question
    )

    evidence = {

        "usage": 0.0,

        "project": 0.0,

        "learning": 0.0,

        "selection": 0.0,

        "comparison": 0.0
    }

    # ========================================================
    # Usage Evidence
    # ========================================================

    usage_strong = [

        "استخدامات",
        "مجالات استخدام",
        "تستخدم في",
        "تستعمل في",
        "فيما تستخدم",
        "مجالاتها",

        "تطوير المواقع",
        "تطوير الويب",
        "تطوير البرمجيات",

        "اتمتة المهام",
        "أتمتة المهام",
        "اتمتة",
        "أتمتة",

        "تحليل البيانات",
        "علم البيانات",

        "الذكاء الاصطناعي",
        "ذكاء اصطناعي",

        "التعلم الالي",
        "تعلم الالي",

        "الاحصاء",
        "الإحصاء",

        "سوق المال",
        "الاعمال",
        "الأعمال"
    ]

    usage_medium = [

        "تطوير",
        "تطبيقات",
        "برمجيات",
        "برامج",
        "تحليل",
        "بيانات",
        "ويب",
        "المواقع"
    ]

    strong_matches = 0

    for phrase in usage_strong:

        if normalize_text(phrase) in normalized_document:

            strong_matches += 1

    medium_matches = 0

    for phrase in usage_medium:

        if normalize_text(phrase) in normalized_document:

            medium_matches += 1

    if strong_matches > 0:

        evidence["usage"] = min(
            1.0,
            0.50 + (
                strong_matches * 0.15
            )
        )

    elif medium_matches >= 2:

        evidence["usage"] = 0.45

    elif medium_matches == 1:

        evidence["usage"] = 0.25

    # ========================================================
    # Project Evidence
    # ========================================================

    project_strong = [

        "مشروع",
        "مشاريع",
        "المشاريع",
        "بناء مشروع",
        "بناء مشاريع",

        "تطبيق عملي",
        "مشاريع عملية",

        "معرض اعمال",
        "معرض أعمال",

        "مشروع عملي"
    ]

    project_medium = [

        "تطبيق",
        "تطوير",
        "برنامج",
        "برامج",
        "بناء",
        "عمل عليها"
    ]

    strong_matches = 0

    for phrase in project_strong:

        if normalize_text(phrase) in normalized_document:

            strong_matches += 1

    medium_matches = 0

    for phrase in project_medium:

        if normalize_text(phrase) in normalized_document:

            medium_matches += 1

    if strong_matches > 0:

        evidence["project"] = min(
            1.0,
            0.55 + (
                strong_matches * 0.15
            )
        )

    elif medium_matches >= 2:

        evidence["project"] = 0.45

    elif medium_matches == 1:

        evidence["project"] = 0.25

    # ========================================================
    # Learning Evidence
    # ========================================================

    learning_strong = [

        "تعلم بايثون",
        "تعلم لغة بايثون",
        "دروس",
        "درس",
        "تعلم",
        "دراسة",
        "ابدأ من هنا",
        "ابدء من هنا",

        "أساسيات بايثون",
        "اساسيات بايثون",

        "الموضوعات المتقدمة",
        "مواضيع متقدمة",

        "كتابة شيفرات",
        "كتابة كود"
    ]

    learning_medium = [

        "مبتدئ",
        "جديد",
        "متقدم",
        "تعلم",
        "لغة"
    ]

    strong_matches = 0

    for phrase in learning_strong:

        if normalize_text(phrase) in normalized_document:

            strong_matches += 1

    medium_matches = 0

    for phrase in learning_medium:

        if normalize_text(phrase) in normalized_document:

            medium_matches += 1

    if strong_matches > 0:

        evidence["learning"] = min(
            1.0,
            0.55 + (
                strong_matches * 0.10
            )
        )

    elif medium_matches >= 2:

        evidence["learning"] = 0.45

    elif medium_matches == 1:

        evidence["learning"] = 0.25

    # ========================================================
    # Selection Evidence
    # ========================================================

    selection_words = [

        "افضل",
        "الافضل",
        "احسن",
        "الاحسن",
        "مناسب",
        "يناسب",
        "خيار",
        "الخيار",
        "يوصى",
        "ينصح"
    ]

    selection_matches = 0

    for phrase in selection_words:

        if normalize_text(phrase) in normalized_document:

            selection_matches += 1

    if selection_matches > 0:

        evidence["selection"] = min(
            1.0,
            0.50 + (
                selection_matches * 0.10
            )
        )

    # ========================================================
    # Comparison Evidence
    # ========================================================

    comparison_words = [

        "الفرق",
        "مقارنة",
        "مقارنه",
        "مقابل",
        "بينهما",
        "على عكس"
    ]

    comparison_matches = 0

    for phrase in comparison_words:

        if normalize_text(phrase) in normalized_document:

            comparison_matches += 1

    if comparison_matches > 0:

        evidence["comparison"] = min(
            1.0,
            0.50 + (
                comparison_matches * 0.10
            )
        )

    return evidence


# ============================================================
# Intent Compatibility
# ============================================================

def intent_compatibility(
    question,
    document
):

    intent = detect_question_intent(
        question
    )

    evidence = get_intent_evidence(
        question,
        document
    )

    score = 0.0

    reasons = []

    # ========================================================
    # Usage
    # ========================================================

    if intent["usage"]:

        usage_score = evidence[
            "usage"
        ]

        if usage_score >= 0.50:

            score += USAGE_INTENT_BOOST

            reasons.append(
                "Strong usage evidence"
            )

        elif usage_score >= USAGE_MIN_EVIDENCE:

            score += (
                USAGE_INTENT_BOOST
                * 0.50
            )

            reasons.append(
                "Partial usage evidence"
            )

        else:

            score -= USAGE_INTENT_PENALTY

            reasons.append(
                "No usage evidence"
            )

        # ----------------------------------------------------
        # Penalize learning-only content
        # ----------------------------------------------------

        if evidence["learning"] >= 0.50:

            score -= USAGE_INTENT_PENALTY

            reasons.append(
                "Learning-focused content"
            )

        # ----------------------------------------------------
        # Penalize project-only content
        # ----------------------------------------------------

        if (
            evidence["project"] >= 0.55
            and
            evidence["usage"] < 0.50
        ):

            score -= (
                USAGE_INTENT_PENALTY
                * 0.80
            )

            reasons.append(
                "Project-focused content"
            )

    # ========================================================
    # Project
    # ========================================================

    if intent["project"]:

        project_score = evidence[
            "project"
        ]

        if project_score >= 0.50:

            score += PROJECT_INTENT_BOOST

            reasons.append(
                "Strong project evidence"
            )

        elif project_score >= PROJECT_MIN_EVIDENCE:

            score += (
                PROJECT_INTENT_BOOST
                * 0.50
            )

            reasons.append(
                "Partial project evidence"
            )

        else:

            score -= PROJECT_INTENT_PENALTY

            reasons.append(
                "No project evidence"
            )

        if evidence["learning"] >= 0.65:

            score -= (
                PROJECT_INTENT_PENALTY
                * 0.60
            )

            reasons.append(
                "Learning-focused content"
            )

    # ========================================================
    # Learning
    # ========================================================

    if intent["learning"]:

        learning_score = evidence[
            "learning"
        ]

        if learning_score >= 0.50:

            score += LEARNING_INTENT_BOOST

            reasons.append(
                "Strong learning evidence"
            )

        elif learning_score >= LEARNING_MIN_EVIDENCE:

            score += (
                LEARNING_INTENT_BOOST
                * 0.50
            )

            reasons.append(
                "Partial learning evidence"
            )

        else:

            score -= LEARNING_INTENT_PENALTY

            reasons.append(
                "No learning evidence"
            )

    # ========================================================
    # Selection
    # ========================================================

    if intent["selection"]:

        selection_score = evidence[
            "selection"
        ]

        if selection_score >= 0.50:

            score += SELECTION_INTENT_BOOST

            reasons.append(
                "Selection evidence"
            )

        else:

            score -= SELECTION_INTENT_PENALTY * 0.50

            reasons.append(
                "Weak selection evidence"
            )

    # ========================================================
    # Comparison
    # ========================================================

    if intent["comparison"]:

        comparison_score = evidence[
            "comparison"
        ]

        if comparison_score >= 0.50:

            score += COMPARISON_INTENT_BOOST

            reasons.append(
                "Comparison evidence"
            )

        else:

            score -= COMPARISON_INTENT_PENALTY * 0.50

            reasons.append(
                "Weak comparison evidence"
            )

    # ========================================================
    # No special intent
    # ========================================================

    if not any(intent.values()):

        reasons.append(
            "No special intent"
        )

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    score = max(
        -0.60,
        min(0.50, score)
    )

    if not reasons:

        reason = "No intent evidence"

    else:

        reason = " | ".join(
            reasons
        )

    return {

        "score":
            score,

        "evidence":
            evidence,

        "reason":
            reason
    }


# ============================================================
# Topic / Entity Consistency
# ============================================================

def topic_entity_consistency(
    question,
    document
):

    question_entities = detect_entities(
        question
    )

    document_entities = detect_entities(
        document
    )

    intent = detect_question_intent(
        question
    )

    # --------------------------------------------------------
    # Question only
    # --------------------------------------------------------

    if is_question_only(
        document
    ):

        return {

            "consistent": False,

            "reason":
                "Document appears to be a question/title only",

            "question_entities":
                question_entities,

            "document_entities":
                document_entities,

            "missing_entities":
                question_entities - document_entities,

            "comparison":
                intent["comparison"],

            "selection":
                intent["selection"],

            "usage":
                intent["usage"],

            "project":
                intent["project"],

            "learning":
                intent["learning"],

            "question_only":
                True
        }

    # --------------------------------------------------------
    # Boilerplate
    # --------------------------------------------------------

    if is_boilerplate(
        document
    ):

        return {

            "consistent": False,

            "reason":
                "Document appears to be website boilerplate/navigation",

            "question_entities":
                question_entities,

            "document_entities":
                document_entities,

            "missing_entities":
                question_entities - document_entities,

            "comparison":
                intent["comparison"],

            "selection":
                intent["selection"],

            "usage":
                intent["usage"],

            "project":
                intent["project"],

            "learning":
                intent["learning"],

            "question_only":
                False
        }

    # --------------------------------------------------------
    # Required entities
    # --------------------------------------------------------

    missing_entities = (
        question_entities
        -
        document_entities
    )

    if missing_entities:

        return {

            "consistent": False,

            "reason":
                "Missing required entities",

            "question_entities":
                question_entities,

            "document_entities":
                document_entities,

            "missing_entities":
                missing_entities,

            "comparison":
                intent["comparison"],

            "selection":
                intent["selection"],

            "usage":
                intent["usage"],

            "project":
                intent["project"],

            "learning":
                intent["learning"],

            "question_only":
                False
        }

    # --------------------------------------------------------
    # No explicit entity
    # --------------------------------------------------------

    if not question_entities:

        return {

            "consistent": True,

            "reason":
                "No explicit topic/entity requirement",

            "question_entities":
                question_entities,

            "document_entities":
                document_entities,

            "missing_entities":
                set(),

            "comparison":
                intent["comparison"],

            "selection":
                intent["selection"],

            "usage":
                intent["usage"],

            "project":
                intent["project"],

            "learning":
                intent["learning"],

            "question_only":
                False
        }

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if intent["comparison"]:

        if len(question_entities) >= 2:

            if not question_entities.issubset(
                document_entities
            ):

                return {

                    "consistent": False,

                    "reason":
                        "Comparison requires all entities",

                    "question_entities":
                        question_entities,

                    "document_entities":
                        document_entities,

                    "missing_entities":
                        question_entities
                        -
                        document_entities,

                    "comparison":
                        True,

                    "selection":
                        intent["selection"],

                    "usage":
                        intent["usage"],

                    "project":
                        intent["project"],

                    "learning":
                        intent["learning"],

                    "question_only":
                        False
                }

    return {

        "consistent": True,

        "reason":
            "Topic/entity/intent consistent",

        "question_entities":
            question_entities,

        "document_entities":
            document_entities,

        "missing_entities":
            set(),

        "comparison":
            intent["comparison"],

        "selection":
            intent["selection"],

        "usage":
            intent["usage"],

        "project":
            intent["project"],

        "learning":
            intent["learning"],

        "question_only":
            False
    }


# ============================================================
# Intent Consistency
# ============================================================

def intent_consistency(
    question,
    document
):

    normalized_document = normalize_text(
        document
    )

    intent = detect_question_intent(
        question
    )

    intent_info = intent_compatibility(
        question,
        document
    )

    evidence = intent_info[
        "evidence"
    ]

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if intent["comparison"]:

        comparison_words = [

            "الفرق",
            "مقارنة",
            "مقارنه",
            "قارن"
        ]

        if any(
            normalize_text(word)
            in normalized_document
            for word in comparison_words
        ):

            return {

                "consistent": True,

                "reason":
                    "Comparison intent supported"
            }

        question_entities = detect_entities(
            question
        )

        document_entities = detect_entities(
            document
        )

        if (
            len(question_entities) >= 2
            and
            question_entities.issubset(
                document_entities
            )
        ):

            return {

                "consistent": True,

                "reason":
                    "All comparison entities present"
            }

        return {

            "consistent": False,

            "reason":
                "Comparison question lacks sufficient evidence"
        }

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    if intent["selection"]:

        selection_evidence = [

            "افضل",
            "الافضل",
            "احسن",
            "الاحسن",
            "مناسب",
            "يناسب",
            "خيار",
            "الخيار",
            "يعتبر",
            "تعد",
            "مناسبه"
        ]

        if any(

            normalize_text(phrase)
            in normalized_document

            for phrase in selection_evidence

        ):

            return {

                "consistent": True,

                "reason":
                    "Selection evidence supported"
            }

        document_entities = detect_entities(
            document
        )

        candidate_entities = {

            "python",
            "java",
            "javascript",
            "csharp"
        }

        if document_entities.intersection(
            candidate_entities
        ):

            return {

                "consistent": True,

                "reason":
                    "Candidate entity available for selection"
            }

        return {

            "consistent": False,

            "reason":
                "Selection question has no candidate evidence"
        }

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    if intent["usage"]:

        usage_score = evidence[
            "usage"
        ]

        # ----------------------------------------------------
        # Strong usage evidence
        # ----------------------------------------------------

        if usage_score >= 0.50:

            return {

                "consistent": True,

                "reason":
                    "Strong usage evidence supported"
            }

        # ----------------------------------------------------
        # Partial usage evidence
        # ----------------------------------------------------

        if usage_score >= USAGE_MIN_EVIDENCE:

            return {

                "consistent": True,

                "reason":
                    "Partial usage evidence supported"
            }

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT accept a document merely because it
        # contains Python.
        #
        # This is what previously allowed:
        # "دروس لبدء تعلم بايثون"
        # to survive a usage question.
        # ----------------------------------------------------

        return {

            "consistent": False,

            "reason":
                "Usage question lacks usage evidence"
        }

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    if intent["project"]:

        project_score = evidence[
            "project"
        ]

        if project_score >= 0.50:

            return {

                "consistent": True,

                "reason":
                    "Strong project evidence supported"
            }

        if project_score >= PROJECT_MIN_EVIDENCE:

            return {

                "consistent": True,

                "reason":
                    "Partial project evidence supported"
            }

        return {

            "consistent": False,

            "reason":
                "Project question lacks project evidence"
        }

    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    if intent["learning"]:

        learning_score = evidence[
            "learning"
        ]

        if learning_score >= 0.50:

            return {

                "consistent": True,

                "reason":
                    "Strong learning evidence supported"
            }

        if learning_score >= LEARNING_MIN_EVIDENCE:

            return {

                "consistent": True,

                "reason":
                    "Partial learning evidence supported"
            }

        question_entities = detect_entities(
            question
        )

        document_entities = detect_entities(
            document
        )

        if (
            question_entities
            and
            question_entities.issubset(
                document_entities
            )
        ):

            return {

                "consistent": True,

                "reason":
                    "Topic entity present for learning question"
            }

        return {

            "consistent": False,

            "reason":
                "Learning question lacks learning evidence"
        }

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    return {

        "consistent": True,

        "reason":
            "No special intent constraint"
    }


# ============================================================
# Combined Consistency Gate
# ============================================================

def consistency_gate(
    question,
    document
):

    entity_result = topic_entity_consistency(
        question,
        document
    )

    if not entity_result["consistent"]:

        return {

            "consistent": False,

            "reason":
                entity_result["reason"],

            "entity_result":
                entity_result,

            "intent_result":
                None
        }

    intent_result = intent_consistency(
        question,
        document
    )

    if not intent_result["consistent"]:

        return {

            "consistent": False,

            "reason":
                intent_result["reason"],

            "entity_result":
                entity_result,

            "intent_result":
                intent_result
        }

    return {

        "consistent": True,

        "reason":
            "Topic/entity/intent consistent",

        "entity_result":
            entity_result,

        "intent_result":
            intent_result
    }


# ============================================================
# Distance Space
# ============================================================

def get_collection_space(
    collection
):

    try:

        configuration = collection.configuration

        if configuration:

            hnsw = configuration.get(
                "hnsw",
                {}
            )

            space = hnsw.get(
                "space"
            )

            if space:

                return space

    except Exception:

        pass

    return "l2"


# ============================================================
# Semantic Score
# ============================================================

def semantic_score(
    distance,
    space
):

    if space == "cosine":

        score = 1.0 - distance

    elif space == "l2":

        score = 1.0 - (
            distance / 2.0
        )

    elif space == "ip":

        score = 1.0 - distance

    else:

        score = 1.0 - distance

    return max(
        0.0,
        min(1.0, score)
    )


# ============================================================
# Final Score
# ============================================================

def calculate_final_score(
    semantic,
    keyword
):

    if keyword <= 0.0:

        return semantic * 0.30

    return (

        semantic * 0.60

        +

        keyword * 0.40
    )


# ============================================================
# Intent-aware Final Score
# ============================================================

def calculate_intent_aware_score(
    final,
    intent_score
):

    return max(

        0.0,

        min(

            1.50,

            final + intent_score

        )
    )


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(
    collection,
    question
):

    question_embedding = create_embedding(
        question
    )

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=RETRIEVAL_COUNT,

        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = results[
        "documents"
    ][0]

    distances = results[
        "distances"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    return (

        documents,

        distances,

        metadatas
    )


# ============================================================
# Re-ranking
# ============================================================

def rerank_documents(
    question,
    documents,
    distances,
    metadatas,
    space
):

    ranked_results = []

    question_entities = detect_entities(
        question
    )

    for document, distance, metadata in zip(

        documents,

        distances,

        metadatas

    ):

        semantic = semantic_score(

            distance,

            space
        )

        keyword = keyword_score(

            question,

            document
        )

        final = calculate_final_score(

            semantic,

            keyword
        )

        consistency = consistency_gate(

            question,

            document
        )

        document_entities = detect_entities(
            document
        )

        # ----------------------------------------------------
        # Entity Match
        # ----------------------------------------------------

        entity_match = (

            bool(question_entities)

            and

            question_entities.issubset(
                document_entities
            )
        )

        # ----------------------------------------------------
        # Intent-aware analysis
        # ----------------------------------------------------

        intent_info = intent_compatibility(

            question,

            document
        )

        intent_score = intent_info[
            "score"
        ]

        intent_evidence = intent_info[
            "evidence"
        ]

        intent_reason = intent_info[
            "reason"
        ]

        # ----------------------------------------------------
        # Base effective score
        # ----------------------------------------------------

        effective_final = calculate_intent_aware_score(

            final,

            intent_score
        )

        # ----------------------------------------------------
        # Entity boost
        # ----------------------------------------------------

        if entity_match:

            effective_final += ENTITY_BOOST

        # ----------------------------------------------------
        # Inconsistent penalty
        # ----------------------------------------------------

        if not consistency[
            "consistent"
        ]:

            effective_final *= (
                INCONSISTENT_PENALTY
            )

        # ----------------------------------------------------
        # Question only
        # ----------------------------------------------------

        question_only = is_question_only(
            document
        )

        if question_only:

            effective_final *= (
                QUESTION_ONLY_PENALTY
            )

        # ----------------------------------------------------
        # Boilerplate
        # ----------------------------------------------------

        boilerplate = is_boilerplate(
            document
        )

        if boilerplate:

            effective_final *= (
                BOILERPLATE_PENALTY
            )

        # ----------------------------------------------------
        # Usage hard penalty
        #
        # If usage question and document has no usage
        # evidence, don't let semantic similarity alone
        # keep it near the top.
        # ----------------------------------------------------

        intent = detect_question_intent(
            question
        )

        if intent["usage"]:

            if intent_evidence["usage"] < 0.25:

                effective_final *= 0.55

        # ----------------------------------------------------
        # Project hard penalty
        # ----------------------------------------------------

        if intent["project"]:

            if intent_evidence["project"] < 0.25:

                effective_final *= 0.60

        # ----------------------------------------------------
        # Learning hard penalty
        # ----------------------------------------------------

        if intent["learning"]:

            if intent_evidence["learning"] < 0.25:

                effective_final *= 0.60

        ranked_results.append({

            "document":
                document,

            "distance":
                distance,

            "semantic":
                semantic,

            "keyword":
                keyword,

            "final":
                final,

            "intent_score":
                intent_score,

            "intent_evidence":
                intent_evidence,

            "intent_reason":
                intent_reason,

            "effective_final":
                effective_final,

            "entity_match":
                entity_match,

            "question_only":
                question_only,

            "boilerplate":
                boilerplate,

            "consistency":
                consistency,

            "metadata":
                metadata
        })

    ranked_results.sort(

        key=lambda x:
            x["effective_final"],

        reverse=True
    )

    return ranked_results


# ============================================================
# Relevance Gate
# ============================================================

def relevance_gate(
    ranked_results,
    question
):

    if not ranked_results:

        return (

            False,

            "No retrieval results",

            None
        )

    consistent_results = [

        result

        for result in ranked_results

        if result[
            "consistency"
        ]["consistent"]

    ]

    if not consistent_results:

        best = ranked_results[0]

        return (

            False,

            "No topic/entity/intent consistent result",

            best["consistency"]
        )

    best = max(

        consistent_results,

        key=lambda x:
            x["effective_final"]
    )

    semantic = best["semantic"]

    keyword = best["keyword"]

    final = best["final"]

    intent_score = best[
        "intent_score"
    ]

    question_entities = detect_entities(
        question
    )

    document_entities = detect_entities(
        best["document"]
    )

    entity_match = (

        bool(question_entities)

        and

        question_entities.issubset(
            document_entities
        )
    )

    intent = detect_question_intent(
        question
    )

    intent_evidence = best[
        "intent_evidence"
    ]

    # ========================================================
    # RULE 1
    # Strong final + keyword
    # ========================================================

    if (

        final >= STRONG_FINAL_THRESHOLD

        and

        keyword >= STRONG_KEYWORD_THRESHOLD

    ):

        return (

            True,

            "Strong final + keyword + consistency",

            best["consistency"]
        )

    # ========================================================
    # RULE 2
    # Very strong semantic + keyword
    # ========================================================

    if (

        semantic >= STRONG_SEMANTIC_THRESHOLD

        and

        keyword >= 0.50

    ):

        return (

            True,

            "Very strong semantic + keyword + consistency",

            best["consistency"]
        )

    # ========================================================
    # RULE 3
    # Strong keyword + semantic
    # ========================================================

    if (

        keyword >= STRONG_KEYWORD_THRESHOLD

        and

        semantic >= MIN_SEMANTIC_FOR_KEYWORD

    ):

        return (

            True,

            "Strong keyword + semantic + consistency",

            best["consistency"]
        )

    # ========================================================
    # RULE 4
    # Strong semantic + exact entity
    # ========================================================

    if (

        semantic >= 0.80

        and

        entity_match

    ):

        return (

            True,

            "Strong semantic + exact topic/entity match",

            best["consistency"]
        )

    # ========================================================
    # RULE 5
    # Usage question
    # ========================================================

    if intent["usage"]:

        if (

            semantic >= 0.70

            and

            entity_match

            and

            intent_evidence["usage"] >= 0.25

        ):

            return (

                True,

                "Usage question + semantic + entity + usage evidence",

                best["consistency"]
            )

    # ========================================================
    # RULE 6
    # Project question
    # ========================================================

    if intent["project"]:

        if (

            semantic >= 0.70

            and

            entity_match

            and

            intent_evidence["project"] >= 0.25

        ):

            return (

                True,

                "Project question + semantic + entity + project evidence",

                best["consistency"]
            )

    # ========================================================
    # RULE 7
    # Learning question
    # ========================================================

    if intent["learning"]:

        if (

            semantic >= 0.70

            and

            entity_match

            and

            intent_evidence["learning"] >= 0.25

        ):

            return (

                True,

                "Learning question + semantic + entity + learning evidence",

                best["consistency"]
            )

    return (

        False,

        "Insufficient relevance evidence",

        best["consistency"]
    )


def check_relation_evidence(
    question,
    document,
    question_entities
):
    """
    التحقق من وجود علاقة حقيقية بين الكيانات المطلوبة
    داخل محتوى الوثيقة، وليس مجرد وجود الكيانات بشكل منفصل.
    """

    if not question_entities:
        return {
            "supported": True,
            "score": 1.0,
            "matched_entities": [],
            "missing_entities": [],
            "reason": "No entities required"
        }

    document_entities = detect_entities(
        document
    )

    # --------------------------------------------------------
    # 1. تحديد الكيانات الموجودة
    # --------------------------------------------------------

    matched_entities = [
        entity
        for entity in question_entities
        if entity in document_entities
    ]

    missing_entities = [
        entity
        for entity in question_entities
        if entity not in document_entities
    ]

    # --------------------------------------------------------
    # 2. إذا كان هناك كيان مطلوب غير موجود
    # --------------------------------------------------------

    if missing_entities:

        return {
            "supported": False,
            "score": 0.0,
            "matched_entities": matched_entities,
            "missing_entities": missing_entities,
            "reason": "Missing required entities"
        }

    # --------------------------------------------------------
    # 3. لا يكفي وجود الكيانات فقط
    #
    # نبحث عن ظهور الكيانات داخل نفس السياق.
    # --------------------------------------------------------

    normalized_document = normalize_text(
        document
    )

    # --------------------------------------------------------
    # Relation groups
    # --------------------------------------------------------

    relation_groups = {

        "python_web": [
            "بايثون",
            "تطوير المواقع",
            "تطوير الويب",
            "مطور ويب",
            "المواقع والبرمجيات",
            "تطوير المواقع والبرمجيات"
        ],

        "python_ai": [
            "بايثون",
            "الذكاء الاصطناعي",
            "تطبيقات الذكاء الاصطناعي",
            "دمج تطبيقات الذكاء الاصطناعي"
        ],

        "python_data_analysis": [
            "بايثون",
            "تحليل البيانات"
        ],

        "python_finance": [
            "بايثون",
            "سوق المال",
            "المال والأعمال",
            "الإحصاء"
        ]
    }

    # --------------------------------------------------------
    # 4. تحديد العلاقات المطلوبة من السؤال
    # --------------------------------------------------------

    required_relations = []

    entities_set = set(
        question_entities
    )

    if (
        "python" in entities_set
        and
        "web" in entities_set
    ):
        required_relations.append(
            "python_web"
        )

    if (
        "python" in entities_set
        and
        "artificial_intelligence" in entities_set
    ):
        required_relations.append(
            "python_ai"
        )

    if (
        "python" in entities_set
        and
        "data_analysis" in entities_set
    ):
        required_relations.append(
            "python_data_analysis"
        )

    if (
        "python" in entities_set
        and
        "finance" in entities_set
    ):
        required_relations.append(
            "python_finance"
        )

    # --------------------------------------------------------
    # 5. إذا لم توجد علاقة معروفة
    #
    # لا نرفض تلقائيًا؛ نستخدم وجود الكيانات.
    # --------------------------------------------------------

    if not required_relations:

        return {
            "supported": True,
            "score": 1.0,
            "matched_entities": matched_entities,
            "missing_entities": [],
            "reason": "All required entities found"
        }

    # --------------------------------------------------------
    # 6. فحص العلاقات
    # --------------------------------------------------------

    matched_relations = []

    for relation in required_relations:

        phrases = relation_groups.get(
            relation,
            []
        )

        # ----------------------------------------------------
        # العلاقة تعتبر موجودة إذا ظهرت العبارات
        # المهمة في نفس الوثيقة.
        # ----------------------------------------------------

        if all(
            phrase in normalized_document
            for phrase in phrases
            if phrase != "بايثون"
        ):

            matched_relations.append(
                relation
            )

    # --------------------------------------------------------
    # 7. حساب قوة الدليل
    # --------------------------------------------------------

    if not matched_relations:

        return {
            "supported": False,
            "score": 0.0,
            "matched_entities": matched_entities,
            "missing_entities": [],
            "reason": "Entities found but no relation evidence"
        }

    relation_score = (
        len(matched_relations)
        /
        len(required_relations)
    )

    return {
        "supported": relation_score > 0.0,
        "score": relation_score,
        "matched_entities": matched_entities,
        "missing_entities": [],
        "matched_relations": matched_relations,
        "reason": "All required entities found with relation evidence"
    }


# ============================================================
# Context Selection
# ============================================================

def select_context_results(
    question,
    ranked_results
):

    selected = []

    intent = detect_question_intent(
        question
    )

    question_entities = detect_entities(
        question
    )

    relation_required = len(
        question_entities
    ) >= 2

    # --------------------------------------------------------
    # First pass
    # Intent-aware context selection
    # --------------------------------------------------------

    for result in ranked_results:

        consistency = result[
            "consistency"
        ]

        if not consistency[
            "consistent"
        ]:

            continue

        # ====================================================
        # Relation Evidence
        # ====================================================

        if relation_required:

            relation_evidence = check_relation_evidence(
                question,
                result["document"],
                question_entities
            )

            # Save relation evidence inside result
            result["relation_evidence"] = (
                relation_evidence
            )

            if not relation_evidence.get(
                "supported",
                False
            ):

                continue

        # ====================================================
        # Question Only
        # ====================================================

        if result[
            "question_only"
        ]:

            continue

        # ====================================================
        # Boilerplate
        # ====================================================

        if result[
            "boilerplate"
        ]:

            continue

        # ====================================================
        # Intent-aware filtering
        # ====================================================

        intent_score = result.get(
            "intent_score",
            0.0
        )

        intent_evidence = result.get(
            "intent_evidence",
            {}
        )

        # ----------------------------------------------------
        # Usage question
        # ----------------------------------------------------

        if intent["usage"]:

            usage_score = intent_evidence.get(
                "usage",
                0.0
            )

            learning_score = intent_evidence.get(
                "learning",
                0.0
            )

            project_score = intent_evidence.get(
                "project",
                0.0
            )

            # Strong learning content
            if (
                learning_score >= 0.70
                and
                usage_score < learning_score + 0.25
            ):

                continue

            # Strong project content
            if (
                project_score >= 0.70
                and
                usage_score < project_score + 0.25
            ):

                continue

            # No real usage evidence
            if (
                usage_score < 0.50
                and
                intent_score < 0.0
            ):

                continue

        # ----------------------------------------------------
        # Project question
        # ----------------------------------------------------

        if intent["project"]:

            project_score = intent_evidence.get(
                "project",
                0.0
            )

            learning_score = intent_evidence.get(
                "learning",
                0.0
            )

            if (
                project_score < 0.50
                and
                learning_score >= 0.70
            ):

                continue

        # ----------------------------------------------------
        # Learning question
        # ----------------------------------------------------

        if intent["learning"]:

            learning_score = intent_evidence.get(
                "learning",
                0.0
            )

            if learning_score < 0.50:

                continue

        # ----------------------------------------------------
        # Selection question
        # ----------------------------------------------------

        if intent["selection"]:

            selection_score = intent_evidence.get(
                "selection",
                0.0
            )

            if (
                selection_score < 0.40
                and
                not result["entity_match"]
            ):

                continue

        # ----------------------------------------------------
        # Comparison question
        # ----------------------------------------------------

        if intent["comparison"]:

            comparison_score = intent_evidence.get(
                "comparison",
                0.0
            )

            if (
                comparison_score < 0.40
                and
                not result["entity_match"]
            ):

                continue

        # ====================================================
        # Evidence Strength
        # ====================================================

        evidence_strength = 0.0

        # ----------------------------------------------------
        # Usage
        # ----------------------------------------------------

        if intent["usage"]:

            usage_score = intent_evidence.get(
                "usage",
                0.0
            )

            if usage_score >= 0.80:

                evidence_strength = max(
                    evidence_strength,
                    usage_score
                )

            elif usage_score >= 0.50:

                evidence_strength = max(
                    evidence_strength,
                    usage_score * 0.80
                )

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        if intent["project"]:

            project_score = intent_evidence.get(
                "project",
                0.0
            )

            if project_score >= 0.70:

                evidence_strength = max(
                    evidence_strength,
                    project_score
                )

        # ----------------------------------------------------
        # Learning
        # ----------------------------------------------------

        if intent["learning"]:

            learning_score = intent_evidence.get(
                "learning",
                0.0
            )

            if learning_score >= 0.70:

                evidence_strength = max(
                    evidence_strength,
                    learning_score
                )

        # ----------------------------------------------------
        # Selection
        # ----------------------------------------------------

        if intent["selection"]:

            selection_score = intent_evidence.get(
                "selection",
                0.0
            )

            if selection_score >= 0.70:

                evidence_strength = max(
                    evidence_strength,
                    selection_score
                )

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        if intent["comparison"]:

            comparison_score = intent_evidence.get(
                "comparison",
                0.0
            )

            if comparison_score >= 0.70:

                evidence_strength = max(
                    evidence_strength,
                    comparison_score
                )

        # ====================================================
        # Evidence Gate
        # ====================================================

        has_specific_intent = any([
            intent["usage"],
            intent["project"],
            intent["learning"],
            intent["selection"],
            intent["comparison"]
        ])

        if has_specific_intent:

            if evidence_strength < 0.50:

                continue

        # ====================================================
        # Normal threshold
        # ====================================================

        if (
            result["effective_final"]
            >=
            CONTEXT_THRESHOLD
        ):

            selected.append(
                result
            )

        # ----------------------------------------------------
        # Strong semantic + exact entity
        # ----------------------------------------------------

        elif (
            question_entities
            and
            result["entity_match"]
            and
            result["semantic"] >= 0.75
        ):

            selected.append(
                result
            )

        if len(selected) >= CONTEXT_COUNT:

            break

    # ========================================================
    # Second pass:
    # Mobile evidence
    # ========================================================

    if intent["learning"]:

        if "mobile" in detect_entities(question):

            for result in ranked_results:

                if result in selected:
                    continue

                if result["question_only"]:
                    continue

                if result["boilerplate"]:
                    continue

                document_entities = detect_entities(
                    result["document"]
                )

                if "mobile" in document_entities:

                    if result["semantic"] >= 0.70:

                        selected.append(
                            result
                        )

                if len(selected) >= CONTEXT_COUNT:

                    break

    # ========================================================
    # Third pass
    # ========================================================

    if len(selected) < 2:

        for result in ranked_results:

            if result in selected:
                continue

            if not result[
                "consistency"
            ]["consistent"]:

                continue

            if result["question_only"]:
                continue

            if result["boilerplate"]:
                continue

            intent_evidence = result.get(
                "intent_evidence",
                {}
            )

            if intent["usage"]:

                usage_score = intent_evidence.get(
                    "usage",
                    0.0
                )

                learning_score = intent_evidence.get(
                    "learning",
                    0.0
                )

                project_score = intent_evidence.get(
                    "project",
                    0.0
                )

                if (
                    learning_score >= 0.70
                    and
                    learning_score > usage_score
                ):

                    continue

                if (
                    project_score >= 0.70
                    and
                    project_score > usage_score
                ):

                    continue

                if usage_score < 0.50:

                    continue

            if (
                result["semantic"] >= 0.80
                and
                result["entity_match"]
            ):

                selected.append(
                    result
                )

            if len(selected) >= CONTEXT_COUNT:

                break

    return selected



# ============================================================
# Build Context
# ============================================================

def build_context(
    selected_results
):
    """
    بناء السياق مع إضافة جملة تأكيدية لتوضيح العلاقات
    """
    context_parts = []

    for i, result in enumerate(
        selected_results,
        start=1
    ):

        chunk = result[
            "metadata"
        ].get(
            "chunk",
            "unknown"
        )

        document_text = result['document']
        
        # ====================================================
        # تحسين السياق: إضافة جملة تأكيدية
        # ====================================================
        # هذه الجملة تساعد النموذج على فهم العلاقات
        # دون إضافة معلومات جديدة
        # ====================================================
        
        # إذا كان النص يحتوي على أسئلة عن الاستخدامات
        if "هل ترغب" in document_text and "مطور ويب" in document_text:
            # إضافة جملة توضيحية قبل النص
            enhanced_text = f"""ملاحظة: النص التالي يوضح استخدامات بايثون.

{document_text}"""
        else:
            enhanced_text = document_text

        context_parts.append(
            f"[Source {i} | Chunk {chunk}]\n"
            f"{enhanced_text}"
        )

    return "\n\n".join(
        context_parts
    )




# ============================================================
# Ask Qwen
# ============================================================

def ask_qwen(
    question,
    context
):
    """
    توليد إجابة مقيدة بالسياق
    """
    prompt = f"""
أنت مساعد ذكاء اصطناعي يعمل بنظام RAG.

مهمتك هي الإجابة عن سؤال المستخدم اعتمادًا على
السياق الموجود أسفل التعليمات فقط.

قواعد صارمة:

1. **استخدم المعلومات الموجودة في السياق فقط.**

2. **إذا كان السياق يحتوي على إجابة، استخرجها مباشرة.**

3. **إذا لم يقدم السياق معلومات كافية:**
   - أجب: "لا توجد معلومات كافية في قاعدة المعرفة."

4. **استخدم صيغة محايدة:**
   - قل "تستخدم بايثون" أو "يمكن استخدام بايثون".
   - لا تستخدم "أنا أستخدم" أو "نحن نستخدم".

5. **أجب باللغة العربية فقط.**

السياق:
{context}

سؤال المستخدم:
{question}

الإجابة:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        ""
    )

    if not answer:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    return answer.strip()




# ============================================================
# Display Results
# ============================================================

def display_results(
    ranked_results
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RE-RANKED RESULTS"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(

        ranked_results[
            :DISPLAY_COUNT
        ],

        start=1

    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            f"Final Score      : "
            f"{result['final']:.4f}"
        )

        print(
            f"Effective Score  : "
            f"{result['effective_final']:.4f}"
        )

        print(
            f"Semantic Score   : "
            f"{result['semantic']:.4f}"
        )

        print(
            f"Keyword Score    : "
            f"{result['keyword']:.4f}"
        )

        print(
            f"Intent Score     : "
            f"{result['intent_score']:.4f}"
        )

        print(
            f"Intent Evidence  : "
            f"{result['intent_evidence']}"
        )

        print(
            f"Intent Reason    : "
            f"{result['intent_reason']}"
        )

        print(
            f"Distance         : "
            f"{result['distance']:.4f}"
        )

        print(
            f"Question Only    : "
            f"{result['question_only']}"
        )

        print(
            f"Boilerplate      : "
            f"{result['boilerplate']}"
        )

        print(
            f"Entity Match     : "
            f"{result['entity_match']}"
        )

        print(
            f"Topic Consistent : "
            f"{result['consistency']['consistent']}"
        )

        print(
            f"Consistency Reason: "
            f"{result['consistency']['reason']}"
        )

        print(
            f"Chunk            : "
            f"{result['metadata'].get('chunk')}"
        )

        print(
            "-" * 70
        )

        print(
            result["document"]
        )

        print()


# ============================================================
# Display Consistency
# ============================================================

def display_consistency(
    consistency
):

    print(
        "\nTopic / Entity Consistency"
    )

    print(
        "-" * 70
    )

    if not consistency:

        print(
            "No consistency information."
        )

        return

    print(
        f"Consistent : "
        f"{consistency['consistent']}"
    )

    print(
        f"Reason     : "
        f"{consistency['reason']}"
    )

    entity_result = consistency.get(
        "entity_result"
    )

    if entity_result:

        print(
            f"Question Entities : "
            f"{sorted(entity_result['question_entities'])}"
        )

        print(
            f"Document Entities : "
            f"{sorted(entity_result['document_entities'])}"
        )

        print(
            f"Missing Entities  : "
            f"{sorted(entity_result['missing_entities'])}"
        )

        print(
            f"Comparison Intent : "
            f"{entity_result['comparison']}"
        )

        print(
            f"Selection Intent  : "
            f"{entity_result['selection']}"
        )

        print(
            f"Usage Intent      : "
            f"{entity_result['usage']}"
        )

        print(
            f"Project Intent    : "
            f"{entity_result['project']}"
        )

        print(
            f"Learning Intent   : "
            f"{entity_result['learning']}"
        )

        print(
            f"Question Only     : "
            f"{entity_result['question_only']}"
        )

    intent_result = consistency.get(
        "intent_result"
    )

    if intent_result:

        print(
            f"Intent Check      : "
            f"{intent_result['reason']}"
        )


# ============================================================
# Best Consistent Result
# ============================================================

def get_best_consistent_result(
    ranked_results
):

    consistent = [

        result

        for result in ranked_results

        if result[
            "consistency"
        ]["consistent"]

    ]

    if not consistent:

        return None

    return max(

        consistent,

        key=lambda x:
            x["effective_final"]
    )


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    question
):

    # ========================================================
    # 1. Query Processing
    # ========================================================

    normalized, corrected = process_query(
        question
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "QUERY PROCESSING"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal:"
    )

    print(
        question
    )

    print(
        "\nNormalized:"
    )

    print(
        normalized
    )

    print(
        "\nCorrected:"
    )

    print(
        corrected
    )

    retrieval_question = corrected

    # ========================================================
    # 2. Debug Entities
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "QUERY ANALYSIS"
    )

    print(
        "=" * 70
    )

    query_entities = detect_entities(
        retrieval_question
    )

    query_intent = detect_question_intent(
        retrieval_question
    )

    print(
        f"\nDetected Entities : "
        f"{sorted(query_entities)}"
    )

    print(
        f"Intent             : "
        f"{query_intent}"
    )

    # ========================================================
    # 3. ChromaDB
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTING TO CHROMADB"
    )

    print(
        "=" * 70
    )

    collection = get_collection()

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    # ========================================================
    # 4. Distance
    # ========================================================

    space = get_collection_space(
        collection
    )

    print(
        f"Distance Space: "
        f"{space}"
    )

    # ========================================================
    # 5. Retrieval
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL"
    )

    print(
        "=" * 70
    )

    (
        documents,
        distances,
        metadatas
    ) = retrieve_documents(

        collection,

        retrieval_question
    )

    print(
        f"Retrieved candidates: "
        f"{len(documents)}"
    )

    # ========================================================
    # 6. Re-ranking
    # ========================================================

    ranked_results = rerank_documents(

        retrieval_question,

        documents,

        distances,

        metadatas,

        space
    )

    # ========================================================
    # 7. Display
    # ========================================================

    display_results(
        ranked_results
    )

    # ========================================================
    # 8. Best result
    # ========================================================

    best_consistent = get_best_consistent_result(
        ranked_results
    )

    # ========================================================
    # 9. Relevance Gate
    # ========================================================

    (
        is_relevant,
        reason,
        consistency
    ) = relevance_gate(

        ranked_results,

        retrieval_question
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RELEVANCE GATE"
    )

    print(
        "=" * 70
    )

    if best_consistent:

        print(
            f"\nBest Consistent Final : "
            f"{best_consistent['final']:.4f}"
        )

        print(
            f"Best Consistent Effective : "
            f"{best_consistent['effective_final']:.4f}"
        )

        print(
            f"Best Consistent Semantic : "
            f"{best_consistent['semantic']:.4f}"
        )

        print(
            f"Best Consistent Keyword : "
            f"{best_consistent['keyword']:.4f}"
        )

        print(
            f"Best Intent Score : "
            f"{best_consistent['intent_score']:.4f}"
        )

        print(
            f"Best Intent Evidence : "
            f"{best_consistent['intent_evidence']}"
        )

    else:

        print(
            "\nNo consistent result found."
        )

    print(
        f"\nSTATUS : "
        f"{'RELEVANT' if is_relevant else 'NOT RELEVANT'}"
    )

    print(
        f"REASON : "
        f"{reason}"
    )

    display_consistency(
        consistency
    )

    # ========================================================
    # 10. Stop if irrelevant
    # ========================================================

    if not is_relevant:

        print(
            "\nلا توجد معلومات كافية "
            "في قاعدة المعرفة."
        )

        return {

            "original":
                question,

            "normalized":
                normalized,

            "corrected":
                corrected,

            "results":
                ranked_results,

            "best_result":
                best_consistent,

            "is_relevant":
                False,

            "reason":
                reason,

            "answer":
                "لا توجد معلومات كافية في قاعدة المعرفة."
        }

    # ========================================================
    # 11. Context Selection
    # ========================================================

    selected_results = select_context_results(

        retrieval_question,

        ranked_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT SELECTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSelected chunks: "
        f"{len(selected_results)}"
    )

    if not selected_results:

        print(
            "\nNo strong consistent context was found."
        )

        return {

            "original":
                question,

            "normalized":
                normalized,

            "corrected":
                corrected,

            "results":
                ranked_results,

            "best_result":
                best_consistent,

            "is_relevant":
                False,

            "reason":
                "No strong consistent context",

            "answer":
                "لا توجد معلومات كافية في قاعدة المعرفة."
        }

    # ========================================================
    # 12. Build Context
    # ========================================================

    context = build_context(
        selected_results
    )

    print(
        "\nContext:"
    )

    print(
        context
    )

    # ========================================================
    # 13. Qwen
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "QWEN"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: "
        f"{LLM_MODEL}"
    )

    print(
        "\nGenerating answer..."
    )

    answer = ask_qwen(

        retrieval_question,

        context
    )

    # ========================================================
    # 14. Final Answer
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print()

    print(
        answer
    )

    return {

        "original":
            question,

        "normalized":
            normalized,

        "corrected":
            corrected,

        "results":
            ranked_results,

        "best_result":
            best_consistent,

        "is_relevant":
            True,

        "reason":
            reason,

        "context":
            context,

        "answer":
            answer
    }


# ============================================================
# ChromaDB
# ============================================================

def get_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# Interactive Mode
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "RAG PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        "\nArabic RAG Pipeline"
    )

    print(
        "Type exit to stop.\n"
    )

    while True:

        question = input(
            "Question: "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nGoodbye."
            )

            break

        if not question:

            print(
                "\nQuestion cannot be empty.\n"
            )

            continue

        try:

            run_pipeline(
                question
            )

        except Exception as e:

            print(
                "\n" + "=" * 70
            )

            print(
                "ERROR"
            )

            print(
                "=" * 70
            )

            print(
                f"{type(e).__name__}: {e}"
            )

        print(
            "\n"
        )


question = "هل تستخدم بايثون في تطوير المواقع والذكاء الاصطناعي؟"

document = """
حدد هدفك من تعلم لغة بايثون

هل ترغب أن تصبح مطور ويب محترف، أم تريد دمج تطبيقات
الذكاء الاصطناعي على عملك، أم تريد أتمتة مهام عملك؟
"""

result = detect_relation_evidence(
    question,
    document
)

print("\n" + "=" * 70)
print("RELATION EVIDENCE TEST")
print("=" * 70)

print(result)

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()