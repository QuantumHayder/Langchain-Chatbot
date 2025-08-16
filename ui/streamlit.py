import streamlit as st
from services import chain, history
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

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
        
        response = chain.invocation_series(prompt)
        
        with st.chat_message("assistant"):
        # response =[doc.content for doc in docs]
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content":response})

        history.chat_history.add_messages([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])
