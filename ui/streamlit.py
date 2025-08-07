import streamlit as st
from utils import helper
def chatbot_ask():
    st.title("Chatty Botty")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt:= st.chat_input():
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
   
        response = helper.get_response(prompt)
        
        with st.chat_message("assistant"):
        # response =[doc.content for doc in docs]
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content":response})
    