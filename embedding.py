from google import genai
from google.genai import types
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List, Dict
from dotenv import load_dotenv
import os
import time
import google.genai.errors

load_dotenv()

# Get the actual API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
   
    def __call__(self, input: Documents) -> Embeddings:
      EMBEDDING_MODEL_ID = "gemini-embedding-001"
      title = "Custom query" 
      response = client.models.embed_content(
          model=EMBEDDING_MODEL_ID,
          contents=input,
          config=types.EmbedContentConfig(
            task_type="retrieval_document",
            title=title
          )
      )

      #return response.embeddings[0].values
      return [embedding.values for embedding in response.embeddings]


def create_and_store_embeddings(chunked_documents: list[dict], db: chromadb.Collection) -> None:
    """
    Creates embeddings for document chunks in batches of 100 and stores them in a ChromaDB collection.

    Args:
        chunked_documents (list[dict]): A list of dictionaries, where each dict
                                        has an 'id' and 'text' for a chunk.
        db (chromadb.Collection): The ChromaDB collection to store the embeddings.

    Returns:
        None
    """

    # Extract the texts and ids from your chunked_documents list
    texts_to_embed = [chunk["text"] for chunk in chunked_documents]
    ids_for_db = [chunk["id"] for chunk in chunked_documents]

    # Let's keep the batch size small to be safe.
    batch_size = 20
    total = len(texts_to_embed)
    print(f"\n==== Adding {total} chunks to '{db.name}' with a resilient retry strategy ====")

    # Loop through batches
    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_ids = ids_for_db[i:i+batch_size]
        
        # --- Retry Logic Initialization ---
        max_retries = 4
        retries = 0
        success = False
        initial_delay = 5  # Start with a 5-second delay

        # Keep trying the same batch until it succeeds or max retries are reached
        while not success and retries < max_retries:
            try:
                print(f"--> Attempting to add batch {i//batch_size + 1} (Attempt {retries + 1}/{max_retries})...")
                
                db.add(
                    documents=batch_texts,
                    ids=batch_ids
                )
                
                # If the above line works, we mark as successful and exit the while loop
                print(f"    ✅ Success for batch {i//batch_size + 1}.")
                success = True

            except google.genai.errors.ClientError as e:
                # Specifically catch the 429 Resource Exhausted error
                if e.status == "RESOURCE_EXHAUSTED":
                    retries += 1
                    wait_time = initial_delay * (2 ** retries) # Exponential backoff
                    print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # If it's a different client error, raise it immediately
                    print(f"❌ An unexpected client error occurred: {e}")
                    raise e # Re-raise the exception if it's not a rate limit issue
            
            except Exception as e:
                # Catch any other unexpected errors
                print(f"❌ An unexpected error occurred: {e}")
                raise e # Re-raise it

        if not success:
            print(f"\n🔥🔥🔥 CRITICAL ERROR: Failed to add batch {i//batch_size + 1} after {max_retries} retries. Aborting. 🔥🔥🔥")
            # You might want to break the loop or raise an exception here
            break

    print("\n==== Addition complete. ====")
