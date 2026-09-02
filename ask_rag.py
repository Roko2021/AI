import requests
import chromadb


OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5-coder:7b"

CHROMA_PATH = "./chroma_db"


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

    return response.json()["embeddings"][0]


def ask_qwen(question, context):

    prompt = f"""
أنت مساعد ذكي يعتمد فقط على المعلومات الموجودة في المصدر المرفق.

أجب عن سؤال المستخدم اعتمادًا على المعلومات الموجودة في المصدر.

إذا كانت الإجابة غير موجودة في المصدر، قل بوضوح:
"لا توجد معلومات كافية في المصدر للإجابة عن هذا السؤال."

لا تضف معلومات من خارج المصدر.

المصدر:

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
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


# ---------------------------------------
# Connect to ChromaDB
# ---------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name="website_knowledge"
)


# ---------------------------------------
# Ask question
# ---------------------------------------

question = input("Ask a question: ").strip()


print("\nSearching knowledge base...")


question_embedding = create_embedding(question)


results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


documents = results["documents"][0]


print("\nRetrieved information:\n")

for i, document in enumerate(documents, start=1):

    print("=" * 60)
    print(f"CHUNK {i}")
    print("=" * 60)

    print(document)


# ---------------------------------------
# Build context
# ---------------------------------------

context = "\n\n".join(documents)


print("\nGenerating answer with Qwen...\n")


answer = ask_qwen(
    question,
    context
)


print("=" * 60)
print("QWEN ANSWER")
print("=" * 60)

print(answer)