import streamlit as st
from services import vector_store, history, llm

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
        history.history.add_user_message(prompt)
        
        
    docs = vector_store.vector_store.similarity_search(str(prompt),k=2)
    ai_response = llm.chain.invoke(
        {
            "input": prompt,
            "context": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs]
        }
    )
    
    with st.chat_message("assistant"):
        response = ai_response.content
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    