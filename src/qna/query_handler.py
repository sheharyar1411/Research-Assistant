import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS

# --- MANUAL CHAIN DEFINITIONS  ---

def create_stuff_documents_chain(llm, prompt):
    def format_docs(inputs):
        return "\\n\\n".join([doc.page_content for doc in inputs["context"]])
    return (
        RunnablePassthrough.assign(context=format_docs)
        | prompt
        | llm
        | StrOutputParser()
    )

def create_retrieval_chain(retriever, combine_docs_chain):
    def retrieval_chain_wrapper(inputs):
        docs = retriever.invoke(inputs["input"])
        answer = combine_docs_chain.invoke({"input": inputs["input"], "context": docs})
        return {"answer": answer, "context": docs}
    return retrieval_chain_wrapper

# --- END MANUAL DEFINITIONS ---

#setup for qna system
def qna_system(extracted_texts):

    #breaks documents into chunks
    embedding = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-278m-multilingual")    
    print("Embedding documents into ChromaDB.")

    #retrievel functionality
    vectorstore = Chroma.from_documents(
        documents=extracted_texts, 
        embedding=embedding
    )
    retriever = vectorstore.as_retriever()

    #model 'qwen' used for article qna
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=api_token 
    )

    chat_model = ChatHuggingFace(llm=llm)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\\n\\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    qna_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, qna_chain)

    print("System is ready to answer questions")
    return rag_chain, retriever

#setup for answers
def answer_query(rag_chain, query):

    try:
        results = rag_chain({"input": query})
        answer = results.get("answer", "No answer for this query.")
        context = results.get("context", [])
        return answer, context
    
    except Exception as e:
        return f"An error occured: {e}", []


#setup for followup qna
def followup_qna(query, answer, context_docs):

    print("Generating follow-up questions.")

    try:
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct", 
            task="text-generation",
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=api_token
        )
        
        chat_model_gen = ChatHuggingFace(llm=llm_endpoint)

        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Generate 3 relevant follow-up questions based on the context provided. Return ONLY the questions, one per line, no numbering."),
            ("human", "Original Question: {query} \n Answer: {answer} \n Context: {context}")
        ])
        
        chain = prompt | chat_model_gen | StrOutputParser()
        
        response_str = chain.invoke({
            "query": query,
            "answer": answer,
            "context": context_str[:2000] 
        })
        
        questions = []
        for line in response_str.strip().split("\n"):
            line = line.strip()
            if line and "?" in line:
                if line[0].isdigit() or line[0] in ['-', '*']:
                     for i, char in enumerate(line):
                         if char.isalpha():
                             line = line[i:]
                             break
                questions.append(line)
                
        return questions[:3]
        
    except Exception as e:
        print(f"An error occured: {e}")
        return []


#setup for article critics
def generate_critique(rag_chain):

    print("\nAnalyzing the research...")

    critique_query = (
        "Critically evaluate the research provided in the context. "
        "Focus specifically on: "
        "1. Limitations of the study. "
        "2. Potential biases. "
        "3. What future work is missing? "
        "Provide a skeptical analysis in bullet points."
    )

    try:
        results = rag_chain({"input": critique_query})
        critique = results.get("answer", "Could not generate critics about this article.")
        return critique
    
    except Exception as e:
        return f"An error occured: {e}"


#setup for audio generation
def audio(text, output_file="answer.mp3"):

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, "data")
        
        save_path = os.path.join(data_dir, output_file)
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(save_path)
        
        print("Audio file saved in folder.")
        return save_path
        
    except Exception as e:
        print(f"An error occured: {e}")
        return None
