
# ragbot-from-scratch

A minimal Retrieval-Augmented Generation (RAG) chatbot built from scratch, with no frameworks. Uses Google Gemini for embeddings and ChromaDB for vector storage.

## Features

- Loads PDF, DOCX, and TXT files from a folder
- Splits documents into chunks and embeds them using Gemini
- Stores embeddings in ChromaDB for fast retrieval
- Answers questions using RAG (retrieval-augmented generation) with Gemini

## Setup

1. **Clone the repo:**
	```bash
	git clone <your-repo-url>
	cd ragbot-from-scratch
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Set up your environment variables:**
	- Create a `.env` file in the project root:
	  ```
	  GEMINI_API_KEY=your_google_gemini_api_key
	  ```

4. **Add your data:**
	- Place your PDF, DOCX, or TXT files in the `knowledge_base/` folder (or any folder you specify in the code).

5. **Enable function to build the vector store** 
	- Uncomment the line in `app.py` that calls `create_and_store_embeddings(chunked_documents, vector_db)`.

## Usage

1. **Run the bot:**
	```bash
	python3 app.py
	```
	- The script will load documents, split them into chunks, and (if enabled) create and store embeddings.
	- It will then answer a few sample queries using the knowledge base.

2. **Tweak for your needs:**
	- To use your own data, just add your files to a `knowledge_base` folder.
	- To change the chunk size or overlap, edit the `split_text` function in `app.py`.
	- To change the number of results retrieved, adjust the `n_results` parameter in `query_collection`.
    - You can change the query as well

## Building a Custom Knowledge Base

To build a new knowledge base with your own files:

1. **Delete the existing ChromaDB folder:**
	```bash
	rm -rf chroma_persistent_storage/
	```
2. **Add your new data files** to the data folder (e.g., `finance_books/`).
3. **Uncomment the line** in `app.py` that calls `create_and_store_embeddings(chunked_documents, vector_db)`.. 
4. **Run the script** to rebuild the vector database:
	```bash
	python3 app.py
	```

That’s it! You now have a custom RAG bot with your own knowledge base.
