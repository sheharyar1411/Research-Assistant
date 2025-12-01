import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(directory):
    """
    Load all PDF files from a specified directory.

    Args:
        directory (str): Path to the directory containing PDFs.

    Returns:
        list: List of file paths to the PDF files.
    """
    pdf_files = []
    
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, file))
            
    return pdf_files