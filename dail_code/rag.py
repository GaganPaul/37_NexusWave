import os
import faiss
import numpy as np
import pdfplumber
import pickle
from sentence_transformers import SentenceTransformer

# Directory to store processed documents and embeddings
DOCUMENTS_DIR = "documents"
VECTOR_STORE_PATH = os.path.join(DOCUMENTS_DIR, "faiss_index.bin")
TEXT_STORE_PATH = os.path.join(DOCUMENTS_DIR, "documents.pkl")

# Initialize Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = embedding_model.get_sentence_embedding_dimension()

# Check if existing FAISS index and document store exist
if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(TEXT_STORE_PATH):
    print("Loading existing FAISS index and document store...")
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(TEXT_STORE_PATH, "rb") as f:
        document_store = pickle.load(f)
else:
    print("Initializing new FAISS index and document store...")
    index = faiss.IndexFlatL2(dimension)
    document_store = []


# Function to Process and Store Documents
def process_documents_from_folder():
    """
    Reads all PDF and TXT files in the DOCUMENTS_DIR, vectorizes them, and stores in FAISS index.
    """
    for file_name in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file_name)

        if file_name.endswith((".pdf", ".txt")) and file_path not in document_store:
            print(f"Processing document: {file_path}")
            text = extract_text(file_path)

            if text:
                add_document_to_index(text)
                save_index()
                print(f"Document '{file_name}' stored successfully in FAISS.")
            else:
                print(f"Skipping '{file_name}' (No text found).")


def extract_text(file_path):
    """
    Extracts text from PDF or TXT files.
    """
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text


def add_document_to_index(text):
    """
    Splits the document into chunks, encodes them into vectors, and stores in FAISS index.
    """
    chunks = text.split("\n\n")  # Splitting text into smaller chunks
    embeddings = embedding_model.encode(chunks)

    for chunk, embedding in zip(chunks, embeddings):
        index.add(np.array([embedding], dtype=np.float32))
        document_store.append(chunk)


def save_index():
    """
    Saves the FAISS index and document store persistently.
    """
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(TEXT_STORE_PATH, "wb") as f:
        pickle.dump(document_store, f)


# Function to Retrieve Context
def retrieve_context(query, top_k=3):
    """
    Searches FAISS index for relevant document chunks based on query embedding.
    """
    if not document_store:
        print("No documents available for retrieval.")
        return None

    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k=top_k)

    retrieved_docs = [document_store[i] for i in indices[0] if i < len(document_store)]
    return " ".join(retrieved_docs) if retrieved_docs else None


# Run document processing when script is executed
if __name__ == "__main__":
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)  # Ensure directory exists
    process_documents_from_folder()
