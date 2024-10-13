import streamlit as st
import model

# Set the page configuration
st.set_page_config(page_title="DocAI - Doctor Support", page_icon="🧑‍⚕️", layout="wide")
st.header("DocAI - Doctor Support 🧑‍⚕️")

# Initialize chat history specific to this page
if "messages_chatbot_1" not in st.session_state:
    st.session_state.messages_chatbot_1 = []

# Display chat messages from history on app rerun
for message in st.session_state.messages_chatbot_1:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages_chatbot_1.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    model_response = model.model(prompt, history=st.session_state.messages_chatbot_1)
    
    # Add assistant response to chat history
    st.session_state.messages_chatbot_1.append({"role": "assistant", "content": model_response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(model_response)
