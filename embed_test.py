import ollama

text = "البرمجة بلغة بايثون"

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print(f"Vector dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")