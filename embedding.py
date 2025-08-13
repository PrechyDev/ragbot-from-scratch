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


def create_and_store_embeddings(chunked_documents: List[Dict[str, str]], db: chromadb.Collection) -> None:
    """
    Creates embeddings using a tuned, steady-state approach designed to stay
    within the API's rate limits, while still including a retry mechanism for robustness.
    """
    texts_to_embed = [chunk["text"] for chunk in chunked_documents]
    ids_for_db = [chunk["id"] for chunk in chunked_documents]

    # --- TUNED PARAMETERS BASED ON YOUR LIMITS AND WORKLOAD ---
    # Optimal batch size to stay under the token-per-minute limit.
    batch_size = 10
    # Time to wait between each batch to maintain a steady request rate.
    sleep_time_seconds = 5 
    # -------------------------------------------------------------

    total = len(texts_to_embed)
    print(f"\n==== Starting tuned ingestion of {total} chunks to '{db.name}' ====")
    print(f"     Configuration: {batch_size} chunks every {sleep_time_seconds} seconds.")

    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_ids = ids_for_db[i:i+batch_size]
        
        # --- Retry logic is still kept as a safety net ---
        max_retries = 3
        current_retry = 0
        success = False

        while not success and current_retry < max_retries:
            try:
                print(f"--> Processing batch {i//batch_size + 1}/{total//batch_size + 1}...")
                
                db.add(
                    documents=batch_texts,
                    ids=batch_ids
                )
                
                print(f"    ✅ Success. Pausing for {sleep_time_seconds} seconds...")
                success = True
                # The pause happens *after* a successful request.
                time.sleep(sleep_time_seconds)

            except google.genai.errors.ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    current_retry += 1
                    wait_time = 10 * (2 ** current_retry) # Exponential backoff for safety
                    print(f"    ⚠️ Unexpected rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e # A different error occurred, so we stop.
            
            except Exception as e:
                raise e

        if not success:
            print(f"\n🔥🔥🔥 CRITICAL ERROR: Failed to add batch {i//batch_size + 1} after {max_retries} retries. Aborting. 🔥🔥🔥")
            break

    print("\n==== Tuned ingestion complete. ====")


