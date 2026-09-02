import requests
import chromadb

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
# Test Questions
# ============================================================

RELATED_QUESTIONS = [

    "ما هي أبرز استخدامات بايثون؟",

    "لماذا أتعلم لغة بايثون؟",

    "هل تعلم لغة بايثون سهل؟",

    "ما هي المشاريع التي يمكن العمل عليها عند تعلم بايثون؟",

    "ما هو الوقت المطلوب لتعلم لغة بايثون؟",

    "كيف أتعلم البرمجة بلغة بايثون؟",

]


UNRELATED_QUESTIONS = [

    "ما هو سعر الدولار اليوم؟",

    "ما هي عاصمة فرنسا؟",

    "كيف أطبخ المكرونة؟",

    "ما هو أفضل هاتف في 2026؟",

    "ما حالة الطقس اليوم؟",

    "من هو محمد صلاح؟",

    "كيف أصلح السيارة؟",

    "ما هو سعر الذهب اليوم؟",

]


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
# Normalize Text
# ============================================================

def normalize_text(text):

    import re

    text = text.lower()

    text = re.sub(
        r"[إأآا]",
        "ا",
        text
    )

    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Keywords
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
# Detect Distance Space
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
    # Process Arabic Query
    # --------------------------------------------------------

    normalized, corrected = process_query(
        question
    )

    # --------------------------------------------------------
    # Embedding
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

    ranked.sort(
        key=lambda x: x["final"],
        reverse=True
    )

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
# Main
# ============================================================

def main():

    print("=" * 70)

    print("RELEVANCE EVALUATION")

    print("=" * 70)

    print(
        "\nThis test does NOT call Qwen."
    )

    print(
        "It only evaluates Retrieval + Re-ranking."
    )

    print("\nConnecting to ChromaDB...")

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
    # Build Test Set
    # --------------------------------------------------------

    test_questions = []

    for question in RELATED_QUESTIONS:

        test_questions.append(
            (
                question,
                "RELATED"
            )
        )

    for question in UNRELATED_QUESTIONS:

        test_questions.append(
            (
                question,
                "UNRELATED"
            )
        )

    # --------------------------------------------------------
    # Run Tests
    # --------------------------------------------------------

    results = []

    print("\nRunning evaluation...")

    for index, (
        question,
        expected
    ) in enumerate(
        test_questions,
        start=1
    ):

        print(
            f"\n[{index}/{len(test_questions)}] "
            f"{question}"
        )

        result = evaluate_question(
            collection,
            question,
            expected,
            space
        )

        results.append(
            result
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("SUMMARY")

    print("=" * 70)

    print(
        "\nRELATED QUESTIONS"
    )

    print("-" * 70)

    related_scores = []

    for result in results:

        if result["expected"] == "RELATED":

            related_scores.append(
                result["final"]
            )

            print(
                f"{result['final']:.4f}"
                f" | Semantic="
                f"{result['semantic']:.4f}"
                f" | Keyword="
                f"{result['keyword']:.4f}"
                f" | "
                f"{result['question']}"
            )

    print(
        "\nUNRELATED QUESTIONS"
    )

    print("-" * 70)

    unrelated_scores = []

    for result in results:

        if result["expected"] == "UNRELATED":

            unrelated_scores.append(
                result["final"]
            )

            print(
                f"{result['final']:.4f}"
                f" | Semantic="
                f"{result['semantic']:.4f}"
                f" | Keyword="
                f"{result['keyword']:.4f}"
                f" | "
                f"{result['question']}"
            )

    # --------------------------------------------------------
    # Min / Max
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("SCORE RANGE")

    print("=" * 70)

    if related_scores:

        print(
            "\nRELATED:"
        )

        print(
            f"Minimum Final Score: "
            f"{min(related_scores):.4f}"
        )

        print(
            f"Maximum Final Score: "
            f"{max(related_scores):.4f}"
        )

    if unrelated_scores:

        print(
            "\nUNRELATED:"
        )

        print(
            f"Minimum Final Score: "
            f"{min(unrelated_scores):.4f}"
        )

        print(
            f"Maximum Final Score: "
            f"{max(unrelated_scores):.4f}"
        )

    # --------------------------------------------------------
    # Separation
    # --------------------------------------------------------

    if related_scores and unrelated_scores:

        lowest_related = min(
            related_scores
        )

        highest_unrelated = max(
            unrelated_scores
        )

        print("\n")

        print("=" * 70)

        print("SEPARATION")

        print("=" * 70)

        print(
            f"\nLowest RELATED score : "
            f"{lowest_related:.4f}"
        )

        print(
            f"Highest UNRELATED score: "
            f"{highest_unrelated:.4f}"
        )

        if highest_unrelated < lowest_related:

            suggested = (
                highest_unrelated
                +
                (
                    lowest_related
                    -
                    highest_unrelated
                ) / 2
            )

            print(
                "\nGood separation detected."
            )

            print(
                f"Suggested threshold: "
                f"{suggested:.4f}"
            )

        else:

            print(
                "\nWARNING:"
            )

            print(
                "The score ranges overlap."
            )

            print(
                "We need better relevance signals."
            )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()