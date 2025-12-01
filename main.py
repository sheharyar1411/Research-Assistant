import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
DATA_DIR = os.path.join(project_root, "data")

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxx"

 
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = "lc_xxxxxxx"
    
from src.pdf_processing.pdf_loader import load_pdfs
from src.pdf_processing.text_extractor import extract_text
from src.qna.query_handler import answer_query, qna_system, followup_qna, generate_critique, audio

def main():
    os.environ['LANGCHAIN_TRACING_V2'] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "Auto-Analyst"
    
    print("--------------------------------AUTO ANALYST SYSTEM--------------------------------")
   
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("Exiting due to non availablity of Hugging face API.")
        return

    pdf_files = load_pdfs(DATA_DIR)
    if not pdf_files:
        print(f"No files found.")
        return
    
    print(f"Processing {len(pdf_files)} PDF files...")
    extracted_texts = extract_text(pdf_files)
    rag_chain, retriever = qna_system(extracted_texts)
    
    print("\n--- Q&A system is ready ---")
    print("Commands:")
    print("• Ask any question about the uploaded research article.")
    print("• Type 'critique' to generate the article's critical analysis.")
    print("• Type 'audio' to generate audio of the last response.")
    print("• Type 'exit' to quit.")
    
    last_response = "" #stores the last response
    
    while True:
        user_query = input("\nInput: ")
        
        if user_query.lower() == 'exit':
            print("Exiting the system.")
            break
        
        #for audio
        elif user_query.lower() == 'audio':
            if last_response:
                audio(last_response)
            else:
                print("No previous answer to convert.")
            continue

        #for critics
        elif user_query.lower() == 'critique':
            critique = generate_critique(rag_chain)
            print(f"\n Critical Analysis:\n{critique}")
            last_response = critique
            continue
            
        #for normal qna
        print("Thinking..")
        answer, context = answer_query(rag_chain, user_query)
        print(f"Answer: {answer}")
        last_response = answer
        
        #for followup qna
        questions = followup_qna(user_query, answer, context)

        if questions:
            print("\n--- Follow-up Questions and Answers ---\n")

            for q in questions:
                if not q or q.strip() == "?": continue
                print(f"\nFollow-up Question: {q}")

                f_answer, _ = answer_query(rag_chain, q)
                print(f"Follow-up Answer: {f_answer}")

if __name__ == "__main__":
    main()




# pip install streamlit langchain langchain-community langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers gTTS
# pip install transformers accelerate datasets matplotlib