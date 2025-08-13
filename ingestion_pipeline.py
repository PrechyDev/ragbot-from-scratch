
import chromadb
from embedding import GeminiEmbeddingFunction, create_and_store_embeddings
from loaders import load_documents_from_directory

# Custom Gemini Embedding class
gemini_ef = GeminiEmbeddingFunction()

# Intitalize chromadb with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "finance_collection"
vector_db = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

# Text splitting function
def split_text(text, chunk_size=1000, chunk_overlap=20):
    """
    Splits a given text into overlapping chunks of specified size.

    Args:
        text (str): The input text to be split.
        chunk_size (int, optional): The maximum length of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of overlapping characters between consecutive chunks. Must be smaller than chunk_size. Defaults to 20.

    Raises:
        ValueError: If chunk_overlap is greater than or equal to chunk_size.

    Returns:
        List[str]: A list of non-empty text chunks, each with a maximum length of chunk_size and overlapping by chunk_overlap characters.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:  # Avoid empty chunks
            chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_size - chunk_overlap
    return chunks

# Load documents from the specified directory

directory_path = "knowledge_base/"
documents = load_documents_from_directory(directory_path)

print(f"\nLoaded {len(documents)} documents from '{directory_path}'.")

# Split documents into chunks
chunked_documents = []
print("==== Splitting docs into chunks ====")
for doc in documents:
    print(f"Processing document: {doc['id']}")
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print(f"Created {len(chunked_documents)} chunks from documents.")

# Create and store embeddings in the vector database
create_and_store_embeddings(chunked_documents, vector_db)

