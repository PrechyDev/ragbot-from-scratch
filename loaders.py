import fitz # type: ignore
import pytesseract # type: ignore
from PIL import Image
import io
from docx import Document
import os
from typing import List, Dict, Optional, Any


def load_pdf(file_path: str) -> Optional[str]:
    """
    Extracts text content from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error loading PDF file {file_path}: {e}")
        return None


def ocr_pdf(file_path: str) -> Optional[str]:
    """
    Extracts text from a scanned PDF using OCR.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR on PDF file {file_path}: {e}")
        return None
    

def load_docx(file_path: str) -> Optional[str]:
    """
    Extracts text content from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text content.
    """
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error loading DOCX file {file_path}: {e}")
        return None


def load_text_file(file_path: str) -> Optional[str]:
    """
    Reads content from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None


def load_document(file_path):
    """
    Loads a document from a file path, supporting PDF, DOCX, and TXT.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text content, or None if the format is unsupported or an error occurs.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        # First, try standard text extraction
        content = load_pdf(file_path)
        # If that fails or returns little text, you might try OCR as a fallback
        if not content or len(content.strip()) < 100: # Heuristic to detect scanned PDF
             print("Falling back to OCR for PDF...")
             content = ocr_pdf(file_path)
        return content
    elif file_extension == ".docx":
        return load_docx(file_path)
    elif file_extension == ".txt":
        return load_text_file(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return None


def load_documents_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Loads all supported documents from a specified directory.

    This function iterates through all files in the given directory,
    invokes a generic `load_document` function for each, and collects
    the content of successfully loaded documents.

    Args:
        directory_path (str): The absolute or relative path to the directory
                              containing the documents.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a loaded document and contains its
                               'id' (filename) and 'text' content.
                               Returns an empty list if the directory cannot be
                               accessed or contains no supported files.
    """
    print("==== Loading documents from directory ====")
    documents = []
    
    # Robustly check if the directory exists and is accessible
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return documents

    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Ensure we are only processing files, not subdirectories
            if os.path.isfile(file_path):
                # Use the generic loader, which handles different file types
                content = load_document(file_path)
                
                # Only add the document if content was successfully extracted
                if content:
                    documents.append({"id": filename, "text": content})
                else:
                    print(f"Skipping {filename}: content could not be extracted.")
                    
    except PermissionError:
        print(f"Error: Permission denied to access directory '{directory_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return documents



# Example usage

# loaded_docs = load_documents_from_directory("finance_books/")

# print("\n" + "="*40)
# print("DISPLAYING FIRST 500 CHARACTERS OF EACH DOC")
# print("="*40)

# if loaded_docs:
#     for doc in loaded_docs:
#         doc_name = doc['id']
#         doc_text = doc['text']
            
#         print(f"\n--- Document Name: {doc_name} ---")
#         print(doc_text[:500]) # Slice the text to get the first 500 characters
            
#         if len(doc_text) > 500:
#             print("...") # Add ellipsis to indicate the text was truncated
            
#         print("-" * (len(doc_name) + 20)) # Print a separator
# else:
#     print("\nNo documents were loaded or found.")


# Example usage:
# text_content = load_text_file("path/to/your/file.txt")
# if text_content:
#     print("Successfully loaded content from text file.")

# Example usage:
# docx_content = load_docx("path/to/your/document.docx")
# if docx_content:
#     print("Successfully extracted text from DOCX.")


# Example usage:
# pdf_content = load_pdf("finance_books/The Total Money Makeover - Dave Ramsey.pdf")
# if pdf_content:
#     print("Successfully extracted text from PDF.")
#     print(pdf_content[:500])  # Print first 500 characters