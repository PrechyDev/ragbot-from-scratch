import os
from dotenv import load_dotenv
import chromadb
from google import genai
from ingestion_pipeline import vector_db
from google.genai import types
from typing import List
import asyncio
import time

# --- NEW: Import the reranker library ---
from ragatouille import RAGPretrainedModel

# load env variables and setup embedding function
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

# Initialize the GenAI client
client = genai.Client(api_key=gemini_key)

# save vector_db for use
# vector_db = vector_db

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

# --- NEW: Function for ColBERT Re-ranking ---
def rerank_with_colbert(reranker_model: RAGPretrainedModel, query: str, documents: list[str], k: int = 3) -> List[str]:
    """Re-ranks documents using a ColBERT model and returns the top k results."""
    if not documents:
        return []
    print(f"--> Re-ranking {len(documents)} documents with ColBERT for precision...")
    reranked_results = reranker_model.rerank(query=query, documents=documents)
    # Return the content of the top k ranked documents
    return [doc['content'] for doc in reranked_results[:k]] if reranked_results else []

# Chat engine
chat = client.aio.chats.create(
    model="gemini-2.5-flash", 
    history=[], 
    config=types.GenerateContentConfig(
        system_instruction="You are an expert financial advisor and educator in Nigeria. Always be polite and offer clear, concise solutions. If you don't know the answer, politely state that you cannot assist with that specific query.",
    )
)


async def answer_with_rag(
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
    start_total = time.time()

    # 1. RETRIEVE relevant documents from the database
    start_retrieve = time.time()
    retrieved_chunks = query_collection(
        db=db_collection,
        query=user_query,
        n_results=20 # Retrieve 20 chunks for good context
    )

    # RERANK

    reranked_chunks = rerank_with_colbert(
        reranker_model=RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0"),
        query=user_query,
        documents=retrieved_chunks,
        k=5 # Explicitly ask for the top 5
    )
    print(f"⏱ Retrieval took {time.time() - start_retrieve:.2f} seconds.")

    # 2. AUGMENT: Build the prompt for the LLM
    start_prompt = time.time()
    if not reranked_chunks:
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
        context_string = "\n---\n".join(reranked_chunks)
        
        print(f"✅ Found {len(reranked_chunks)} relevant chunks. Building prompt...")
        
        # This is the RAG prompt template
        prompt = f"""
        You are an expert Nigerian financial advisor. Your knowledge base has been updated with the following highly relevant context. 
        Your task is to synthesize this information to answer the user's question in a clear, natural, and concise way.

        **Instructions:**
        - Integrate the information from the context seamlessly into your response.
        - Do NOT explicitly mention "the context provided" or "the documents." Speak as if this information is your own expertise.
        - Maintain a polite, encouraging, and authoritative Nigerian tone.
        - If the context does not contain the answer, answer based on what your general knowledge and the data you were trained with.

        CONTEXT:
        ---
        {context_string}
        ---

        QUESTION:
        "{user_query}"
        """
    print(f"⏱ Prompt building took {time.time() - start_prompt:.2f} seconds.")
    
    # 3. GENERATE the answer
    # Send the combined prompt to the chat session
    # The `send_message` method handles history automatically
    start_generate = time.time()
    response = await chat_session.send_message(prompt)
    print(f"⏱ Generation took {time.time() - start_generate:.2f} seconds.")

    return response.text

async def main():
    query1 = "What is the conscious spending plan?"
    query2 = "So as a Nigerian students earning arounf 50k a week, how do I start and still live a rich life?"
    query3 = "Can you help me create a sample plan that would work?"
    
    response1 = await answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query1)
    print(response1)

    response2 = await answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query2)
    print(response2)

    response3 = await answer_with_rag(chat_session=chat, db_collection=vector_db, user_query=query3)
    print(response3)

    # for message in chat.get_history():
    #     print(f'role - {message.role}', end=": ")
    #     print(message.parts[0].text)

if __name__ == "__main__":
    asyncio.run(main())



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


