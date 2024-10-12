import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import os

# Load environment variables from the .env file
dotenv.load_dotenv()

# Configure the Google Generative AI API
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return db

def conversation_chain():
    template = """
    You are a helpful assistant for doctor. Now give the details about the test performed on the patient. Do not give any instruction and do not make up the details of the test.
    Context: \n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model_instance, chain_type="stuff", prompt=prompt)
    return chain, model_instance

def user_question(question, db, chain, raw_text, history):
    if db is None:
        st.write("Please upload and process a PDF first.")
        return

    docs = db.similarity_search(question, k=5)  # Retrieve the top 5 most similar documents
    response = chain.invoke(  # Use .invoke() instead of __call__()
        {"input_documents": docs, "question": question, "context": raw_text},
        return_only_outputs=True
    )
    
    final_response = response.get("output_text")  # Adjust this as needed
    return final_response

def main():
    st.set_page_config(page_title="DocAI", page_icon="🧑‍⚕️", layout="wide")
    st.header("DocAI 🧑‍⚕️")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["parts"])

    if question := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "parts": question})
        st.session_state.history.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

    pdf_docs = None
    raw_text = None
    chunks = None
    vector_store = None
    chain = None

    with st.sidebar:
        st.subheader("Menu")
        pdf_docs = st.file_uploader("Upload your Test result in PDF format", accept_multiple_files=True, type="pdf")
        if st.button("Process"):
            if not pdf_docs:
                st.write("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector(chunks)
                chain, model_instance = conversation_chain()

                # Ensure the processing is complete before allowing questions
                if vector_store and chain and raw_text:
                    st.session_state.vector_store = vector_store
                    st.session_state.chain = chain
                    st.session_state.raw_text = raw_text

    if 'vector_store' in st.session_state and 'chain' in st.session_state and 'raw_text' in st.session_state:
        question = "In Bold Letters name the possible disease. And summarize the report"
        response = user_question(question, st.session_state.vector_store, st.session_state.chain, st.session_state.raw_text, st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.session_state.messages.append({"role": "assistant", "parts": response})
            st.markdown(response)  # Display the assistant's response in the main interface

if __name__ == "__main__":
    main()
