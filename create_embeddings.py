from pypdf import PdfReader
import ollama


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


pdf_path = "documents/my_document.pdf"

full_text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    full_text,
    chunk_size=500,
    overlap=100
)

print(f"Number of chunks: {len(chunks)}")

embedded_chunks = []

for index, chunk in enumerate(chunks, start=1):

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    embedded_chunks.append({
        "id": index,
        "text": chunk,
        "embedding": embedding
    })

    print(
        f"Chunk {index}: "
        f"{len(embedding)} dimensions"
    )

print("\nEmbedding process completed.")

print("\nFirst chunk:")
print(embedded_chunks[0]["text"])

print("\nFirst 10 embedding values:")
print(embedded_chunks[0]["embedding"][:10])