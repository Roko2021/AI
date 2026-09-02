import requests
import chromadb
import re

from arabic_query_processor_v2 import process_query


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text:latest"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "website_knowledge_v2"

RETRIEVAL_COUNT = 10


# ============================================================
# Test Groups
# ============================================================

# ------------------------------------------------------------
# 1. Clearly Related
# ------------------------------------------------------------

RELATED_QUESTIONS = [

    "ما هي أبرز استخدامات بايثون؟",

    "لماذا أتعلم لغة بايثون؟",

    "هل تعلم لغة بايثون سهل؟",

    "ما هي المشاريع التي يمكن العمل عليها عند تعلم بايثون؟",

    "ما هو الوقت المطلوب لتعلم لغة بايثون؟",

    "كيف أتعلم البرمجة بلغة بايثون؟",

]


# ------------------------------------------------------------
# 2. Clearly Unrelated
# ------------------------------------------------------------

UNRELATED_QUESTIONS = [

    "ما هو سعر الدولار اليوم؟",

    "ما هي عاصمة فرنسا؟",

    "كيف أطبخ المكرونة؟",

    "ما حالة الطقس اليوم؟",

    "من هو محمد صلاح؟",

    "ما هو سعر الذهب اليوم؟",

]


# ------------------------------------------------------------
# 3. Semantic Traps
#
# These questions contain concepts related to programming
# or Python, but the current knowledge base may not contain
# enough information to answer them.
# ------------------------------------------------------------

SEMANTIC_TRAPS = [

    "ما هي أفضل لغة برمجة لتطوير تطبيقات الهاتف؟",

    "ما الفرق بين بايثون وجافا؟",

    "هل بايثون أفضل من جافا؟",

    "ما هي أفضل لغة برمجة في العالم؟",

    "ما هي أفضل لغة لتحليل البيانات؟",

    "كيف أبدأ مشروع برمجي باستخدام بايثون؟",

    "هل يمكن استخدام بايثون في الشركات؟",

    "ما هي استخدامات جافا؟",

]


# ------------------------------------------------------------
# 4. Partially Related
#
# These are related to Python generally, but the source
# may or may not contain enough information.
# ------------------------------------------------------------

PARTIALLY_RELATED = [

    "هل بايثون مناسبة للمبتدئين؟",

    "كيف أبدأ تعلم بايثون من الصفر؟",

    "هل يمكن استخدام بايثون في الذكاء الاصطناعي؟",

    "هل يمكن استخدام بايثون في تطوير الألعاب؟",

    "هل بايثون مناسبة لتحليل البيانات؟",

    "هل بايثون صعبة في التعلم؟",

]


# ============================================================
# Create Embedding
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
# Normalize Text
# ============================================================

def normalize_text(text):

    text = text.lower()

    # --------------------------------------------------------
    # Arabic Alef normalization
    # --------------------------------------------------------

    text = re.sub(
        r"[إأآا]",
        "ا",
        text
    )

    # --------------------------------------------------------
    # Remove Arabic diacritics
    # --------------------------------------------------------

    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Get Keywords
# ============================================================

def get_keywords(text):

    text = normalize_text(text)

    words = text.split()

    stop_words = {

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
        "كيف",

    }

    return [

        word

        for word in words

        if word not in stop_words
        and len(word) > 1

    ]


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

    matched = 0

    for keyword in question_keywords:

        if keyword in document_text:

            matched += 1

    return matched / len(
        question_keywords
    )


# ============================================================
# Detect Chroma Distance Space
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
# Convert Distance to Semantic Score
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

    # --------------------------------------------------------
    # Important:
    #
    # If no keyword evidence exists, semantic similarity
    # alone should receive only a limited contribution.
    # --------------------------------------------------------

    if keyword == 0.0:

        return semantic * 0.30

    return (
        semantic * 0.60
        +
        keyword * 0.40
    )


# ============================================================
# Evaluate One Question
# ============================================================

def evaluate_question(
    collection,
    question,
    expected,
    space
):

    # --------------------------------------------------------
    # Query Processing
    # --------------------------------------------------------

    normalized, corrected = process_query(
        question
    )

    # --------------------------------------------------------
    # Create Query Embedding
    # --------------------------------------------------------

    question_embedding = create_embedding(
        corrected
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

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

    documents = results["documents"][0]

    distances = results["distances"][0]

    metadatas = results["metadatas"][0]

    # --------------------------------------------------------
    # Re-ranking
    # --------------------------------------------------------

    ranked = []

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
            corrected,
            document
        )

        final = calculate_final_score(
            semantic,
            keyword
        )

        ranked.append({

            "semantic": semantic,

            "keyword": keyword,

            "final": final,

            "distance": distance,

            "document": document,

            "metadata": metadata,

        })

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    ranked.sort(
        key=lambda x: x["final"],
        reverse=True
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best = ranked[0]

    return {

        "question": question,

        "expected": expected,

        "corrected": corrected,

        "semantic": best["semantic"],

        "keyword": best["keyword"],

        "final": best["final"],

        "distance": best["distance"],

        "chunk": best["metadata"].get(
            "chunk"
        ),

        "document": best["document"],

    }


# ============================================================
# Print Result
# ============================================================

def print_result(
    index,
    total,
    result
):

    print()

    print(
        f"[{index}/{total}] "
        f"{result['question']}"
    )

    print(
        f"Expected : "
        f"{result['expected']}"
    )

    print(
        f"Corrected: "
        f"{result['corrected']}"
    )

    print(
        f"Semantic : "
        f"{result['semantic']:.4f}"
    )

    print(
        f"Keyword  : "
        f"{result['keyword']:.4f}"
    )

    print(
        f"Final    : "
        f"{result['final']:.4f}"
    )

    print(
        f"Distance : "
        f"{result['distance']:.4f}"
    )

    print(
        f"Chunk    : "
        f"{result['chunk']}"
    )

    print(
        "Top Document:"
    )

    print(
        result["document"]
    )


# ============================================================
# Print Group Summary
# ============================================================

def print_group_summary(
    results,
    group_name
):

    scores = [
        result["final"]
        for result in results
    ]

    print()

    print("=" * 70)

    print(
        f"{group_name} SUMMARY"
    )

    print("=" * 70)

    print(
        f"Questions: {len(scores)}"
    )

    print(
        f"Minimum Final Score: "
        f"{min(scores):.4f}"
    )

    print(
        f"Maximum Final Score: "
        f"{max(scores):.4f}"
    )

    print(
        f"Average Final Score: "
        f"{sum(scores) / len(scores):.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print("RELEVANCE EVALUATION V2")

    print("=" * 70)

    print(
        "\nThis test does NOT call Qwen."
    )

    print(
        "It evaluates Retrieval + Re-ranking only."
    )

    print(
        "\nPurpose:"
    )

    print(
        "Find questions that can fool semantic search."
    )

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print(
        "\nConnecting to ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    space = get_collection_space(
        collection
    )

    print(
        f"Distance Space: {space}"
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = [

        (
            "RELATED",
            RELATED_QUESTIONS
        ),

        (
            "UNRELATED",
            UNRELATED_QUESTIONS
        ),

        (
            "SEMANTIC TRAPS",
            SEMANTIC_TRAPS
        ),

        (
            "PARTIALLY RELATED",
            PARTIALLY_RELATED
        ),

    ]

    all_results = {}

    total_questions = sum(
        len(questions)
        for _, questions in groups
    )

    current = 0

    # --------------------------------------------------------
    # Run Evaluation
    # --------------------------------------------------------

    for group_name, questions in groups:

        group_results = []

        print()

        print("=" * 70)

        print(
            f"GROUP: {group_name}"
        )

        print("=" * 70)

        for question in questions:

            current += 1

            result = evaluate_question(

                collection,

                question,

                group_name,

                space

            )

            group_results.append(
                result
            )

            print_result(

                current,

                total_questions,

                result

            )

        all_results[group_name] = (
            group_results
        )

        print_group_summary(
            group_results,
            group_name
        )

    # --------------------------------------------------------
    # Global Summary
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("GLOBAL SCORE SUMMARY")

    print("=" * 70)

    for group_name, results in all_results.items():

        scores = [
            result["final"]
            for result in results
        ]

        print()

        print(
            f"{group_name:<20}"
            f" Min={min(scores):.4f}"
            f" Max={max(scores):.4f}"
            f" Avg={sum(scores) / len(scores):.4f}"
        )

    # --------------------------------------------------------
    # Critical Comparisons
    # --------------------------------------------------------

    related_scores = [
        result["final"]
        for result in all_results["RELATED"]
    ]

    unrelated_scores = [
        result["final"]
        for result in all_results["UNRELATED"]
    ]

    trap_scores = [
        result["final"]
        for result in all_results["SEMANTIC TRAPS"]
    ]

    partial_scores = [
        result["final"]
        for result in all_results["PARTIALLY RELATED"]
    ]

    print()

    print("=" * 70)

    print("CRITICAL COMPARISONS")

    print("=" * 70)

    print()

    print(
        f"Lowest RELATED       : "
        f"{min(related_scores):.4f}"
    )

    print(
        f"Highest UNRELATED    : "
        f"{max(unrelated_scores):.4f}"
    )

    print(
        f"Highest SEMANTIC TRAP: "
        f"{max(trap_scores):.4f}"
    )

    print(
        f"Highest PARTIAL      : "
        f"{max(partial_scores):.4f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT change the Relevance Gate yet."
    )

    print(
        "We will use these results to design it."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()