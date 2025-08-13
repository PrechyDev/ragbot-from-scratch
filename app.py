import os
from dotenv import load_dotenv
import chromadb
from google import genai
from chromadb.utils import embedding_functions
from embedding import GeminiEmbeddingFunction, create_and_store_embeddings
from google.genai import types
from loaders import load_documents_from_directory
from typing import List

# load env variables and setup embedding function
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

gemini_ef = GeminiEmbeddingFunction()

# Intitalize chromadb with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "finance_collection"
vector_db = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

# Initialize the GenAI client
client = genai.Client(api_key=gemini_key)

# Function to split text into chunks
def split_text(text, chunk_size=1500, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
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

# print the first item and id
# print("First chunk:")
# print(f"ID: {chunked_documents[0]['id']}")
# print(f"Text: {chunked_documents[0]['text']}")

# Create and store embeddings in the vector database
# Uncomment this if you need to use update the knowledge base.
# create_and_store_embeddings(chunked_documents, vector_db)


# Retrieval
def get_query_embedding(query_text: str) -> List[float]:
    """Generates an embedding for a single query using the correct task type."""
    model_id = "gemini-embedding-001"
    try:
        response = client.models.embed_content(
            model=model_id,
            contents=query_text,  # A single string is passed to 'content'
            config=types.EmbedContentConfig(task_type="retrieval_query") # Correct task type
        )
        # The response for a single content is just one embedding object
        return response.embeddings[0].values
    except Exception as e:
        print(f"An error occurred during query embedding: {e}")
        return []


def query_collection(
    db: chromadb.Collection, query: str, n_results: int = 3
) -> List[str]:
    """
    Queries the ChromaDB collection to retrieve the most relevant document chunks.

    Args:
        db (chromadb.Collection): The collection to query.
        query (str): The user's question.
        n_results (int): The number of relevant chunks to return.

    Returns:
        List[str]: A list of the most relevant document texts.
    """
    print(f"\n==== Retrieving documents for query: '{query}' ====")
    
    # 1. Generate the embedding for the user's query
    query_embedding = get_query_embedding(query)
    
    if not query_embedding:
        print("Could not generate query embedding. Aborting retrieval.")
        return []

    # 2. Query the database using the generated embedding
    results = db.query(
        query_embeddings=[query_embedding],  # Pass the vector, not text
        n_results=n_results
    )
    
    # 3. Return the text of the most relevant documents
    return results['documents'][0]

chat = client.chats.create(
    model="gemini-2.5-flash", 
    history=[], 
    config=types.GenerateContentConfig(
        system_instruction="You are an expert financial advisor and educator in Nigeria. Always be polite and offer clear, concise solutions. If you don't know the answer, politely state that you cannot assist with that specific query.",
    )
)


def answer_with_rag(
    chat_session, 
    db_collection: chromadb.Collection, 
    user_query: str
) -> str:
    """
    Retrieves relevant documents, builds a RAG prompt, and generates an answer.

    Args:
        chat_session (genai.ChatSession): The initialized chat session with the LLM.
        db_collection (chromadb.Collection): The ChromaDB collection with embedded documents.
        user_query (str): The user's original question.

    Returns:
        str: The final, context-aware answer from the LLM.
    """
    # 1. RETRIEVE relevant documents from the database
    retrieved_chunks = query_collection(
        db=db_collection,
        query=user_query,
        n_results=5  # Retrieve 5 chunks for good context
    )
    
    # 2. AUGMENT: Build the prompt for the LLM
    if not retrieved_chunks:
        # If no relevant documents are found, let the LLM know.
        # This respects your system instruction to admit when it doesn't know.
        print("⚠️ No relevant documents found in the database.")
        prompt = f"""
        A user has asked a question, but no relevant context was found in the knowledge base. 
        Please answer the following question based on your general knowledge as a Nigerian financial advisor. 
        Use Nigerian context and tone.
        Remember, if you cannot answer, politely state that to avoid hallucination.

        Question: "{user_query}"
        """
    else:
        # If documents are found, create a context string
        context_string = "\n---\n".join(retrieved_chunks)
        
        print(f"✅ Found {len(retrieved_chunks)} relevant chunks. Building prompt...")
        
        # This is the RAG prompt template
        prompt = f"""
        Based on the following context from financial documents, please provide a clear and concise answer to the user's question as a Nigerian financial advisor.
        Use Nigerian context and tone.

        CONTEXT:
        ---
        {context_string}
        ---

        QUESTION:
        "{user_query}"
        """

    # 3. GENERATE the answer
    # Send the combined prompt to the chat session
    # The `send_message` method handles history automatically
    response = chat_session.send_message(prompt)
    
    return response.text

query1 = "What is investment? How do I start investing?"
query2 = "So as a Nigerian students earning arounf 50k a week, how do I start and still have enough to live on?"
query3 = "What of as a civil worker earning the nigerian minimum wage?"

response1 = answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query1)
print(response1)

response2 = answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query2)
print(response2)

response3 = answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query3)
print(response3)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)



# Example document retrieval
# user_query = "What is investment?"
# retrieved_docs = query_collection(
#     db=vector_db,
#     query=user_query,
#     n_results=3
# )

# print("\n--- Retrieval Results ---")
# for i, doc in enumerate(retrieved_docs):
#     print(f"Result {i+1}: {doc}\n")


