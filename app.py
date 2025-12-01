import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.pdf_processing.pdf_loader import load_pdfs
from src.pdf_processing.text_extractor import extract_text 
from src.qna.query_handler import qna_system, answer_query, followup_qna, generate_critique, audio

st.set_page_config(page_title="Auto Analyst")

st.title("Auto Analyst: Research Assistant")

with st.sidebar:
    st.header("Setup")
    
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxx"

    # os.environ["LANGCHAIN_API_KEY"] = "lc_xxxxxxx"
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_PROJECT"] = "Auto-Analyst"

    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Auto-Analyst"

    st.divider()

    uploaded_files = st.file_uploader("Upload new files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        data_dir = os.path.join(current_dir, "data")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file.")

    st.divider()
    
    if st.button("Load & Process files"):
        with st.spinner("Processing files.."):
            try:
                data_dir = os.path.join(current_dir, "data")
                pdf_files = load_pdfs(data_dir)

                if not pdf_files:
                    st.error(f"No articles found in {data_dir}")
                else:
                    st.success(f"Found {len(pdf_files)} files.")
                    raw_text = extract_text(pdf_files)
                    
                    rag_chain, retriever = qna_system(raw_text)
                    st.session_state['rag_chain'] = rag_chain
                    st.session_state['retriever'] = retriever

                    st.success("System Ready!")

            except Exception as e:
                st.error(f"Error: {e}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the articles uploaded"):

    if 'rag_chain' not in st.session_state:
        st.error("Please load the articles in the sidebar first.")

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            
            if prompt.lower() == "critique":
                response = generate_critique(st.session_state['rag_chain'])
                st.markdown(f"**Critical Analysis:**\n\n{response}")
            
            elif prompt.lower() == "audio":
                if len(st.session_state.messages) > 1:
                    last_assistant_msg = None
                    for msg in reversed(st.session_state.messages[:-1]):
                        if msg["role"] == "assistant":
                            last_assistant_msg = msg["content"]
                            break
                    
                    if last_assistant_msg:
                        audio_path = audio(last_assistant_msg)
                        if audio_path:
                            st.audio(audio_path)
                            response = " "
                        else:
                            response = "Failed to generate audio."
                    else:
                        response = "No previous answer to convert."
                else:
                    response = "No previous answer to convert."
                st.write(response)

            else:
                with st.spinner("Thinking..."):
                    rag_chain = st.session_state['rag_chain']
                    answer, context = answer_query(rag_chain, prompt)
                    st.markdown(answer)
                    response = answer
                    
                    with st.spinner("Generating follow-up questions.."):
                        questions = followup_qna(prompt, answer, context)
                        if questions:
                            st.markdown("---")
                            st.markdown("**Suggested Follow-up Questions and Answers:**")
                            
                            cols = st.columns(len(questions))
                            for idx, q in enumerate(questions):
                                if q:
                                    clean_q = q.replace("**", "").strip()
                                    
                                    st.markdown(f"- {clean_q}")
                                    
                                    followup_ans, _ = answer_query(rag_chain, clean_q)
                                    st.markdown(f"**A:** {followup_ans}")

        st.session_state.messages.append({"role": "assistant", "content": response})

