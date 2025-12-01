from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file_paths):
    """
    Extract text from a list of PDF files using LangChain's PyPDFLoader.

    Args:
        file_paths (list[str]): List of paths to the PDF files.

    Returns:
        list[Document]: A single list containing all text chunks from all documents.
    """
    all_texts = [] 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    for file_path in file_paths:
        if not file_path.endswith(".pdf"):
            print(f"Cant load non pdf files.")
            continue
            
        print(f"Processing file: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        texts = text_splitter.split_documents(docs)
        all_texts.extend(texts)
        
    return all_texts

