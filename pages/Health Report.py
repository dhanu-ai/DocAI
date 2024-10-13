import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import model



# Load environment variables
dotenv.load_dotenv()

gemini_api_key = dotenv.get_key(".env", "GOOGLE_API_KEY")
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# Functions for PDF processing and question answering
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
    return text_splitter.split_text(raw_text)

def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def user_question(question, db, chain, raw_text):
    if db is None:
        return "Please upload and process a PDF first."

    docs = db.similarity_search(question, k=5)
    response = chain.invoke(
        {"input_documents": docs, "question": question, "context": raw_text},
        return_only_outputs=True
    )
    return response.get("output_text")

def conversation_chain():
    template = """
    The scenario is you're a general physician with a medical degree and a patient has arrived with a medical report.
    Now all you have to do is:
    1. Read the medical report.
    2. Answer the question asked by the patient in detail.
    3. Do not make up information.
    4. Use the medical report as the main source.
    5. Highlight the importance of consulting a doctor according to the user's situation.
    6. Focus on {question} and give a relevant answer to it.
    
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt), model_instance

def main():
    # Set the page configuration
    st.set_page_config(page_title="DocAI - Health Report Analysis", page_icon="🧑‍⚕️", layout="wide")
    st.header("DocAI - Health Report Analysis 🧑‍⚕️")

    # Initialize session state variables specific to this page
    if "messages_chatbot_2" not in st.session_state:
        st.session_state.messages_chatbot_2 = []
    if "vector_store_chatbot_2" not in st.session_state:
        st.session_state.vector_store_chatbot_2 = None
    if "chain_chatbot_2" not in st.session_state:
        st.session_state.chain_chatbot_2 = None
    if "raw_text_chatbot_2" not in st.session_state:
        st.session_state.raw_text_chatbot_2 = None

    # Sidebar for PDF file upload and processing
    with st.sidebar:
        st.subheader("Upload Your Test Result")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDF"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector(chunks)
                    chain, _ = conversation_chain()

                    if vector_store and chain and raw_text:
                        st.session_state.vector_store_chatbot_2 = vector_store
                        st.session_state.chain_chatbot_2 = chain
                        st.session_state.raw_text_chatbot_2 = raw_text
                        st.success("PDF processed successfully.")

                        # Initial question for disease identification
                        initial_question = """
                        Your task is:
                        - Name the possible disease according to the test in Bold Letters.
                        - Give some home remedies if possible.
                        - Provide a diet plan to avoid consuming those food items which may affect the disease negatively; the main goal of the diet plan is to minimize the risk of disease.
                        - Give some advice which may help in preventing the disease.
                        """
                        initial_response = user_question(initial_question, vector_store, chain, raw_text)
                        if initial_response:
                            st.session_state.messages_chatbot_2.append({"role": "assistant", "content": initial_response})

    # Display previous messages in the main area
    for message in st.session_state.messages_chatbot_2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user questions
    if st.session_state.vector_store_chatbot_2 and st.session_state.chain_chatbot_2 and st.session_state.raw_text_chatbot_2:
        user_query = st.chat_input("Ask your question:")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages_chatbot_2.append({"role": "user", "content": user_query})

            response = user_question(user_query, st.session_state.vector_store_chatbot_2, st.session_state.chain_chatbot_2, st.session_state.raw_text_chatbot_2)
            if response:
                st.session_state.messages_chatbot_2.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()
