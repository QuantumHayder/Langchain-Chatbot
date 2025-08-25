import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from services.chain import RAGChain
from services.history import ChatHistoryService


class ChatbotUI:
    """
    Streamlit-based UI for the RAG chatbot.
    """

    def __init__(self, rag_chain: RAGChain, chat_history: ChatHistoryService):
        self.rag_chain = rag_chain
        self.chat_history = chat_history

    def run(self):
        st.title("Chatmania: Your Document Chatbot")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input():
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = self.rag_chain.invoke(prompt)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            self.chat_history.add_human_message(prompt)
            self.chat_history.add_ai_message(response)


# def chatbot_ask():
#     st.title("Chatty Botty")
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
        
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#     if prompt:= st.chat_input():
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         st.session_state.messages.append({"role":"user","content":prompt})
        
#         response = chain.invocation_series(prompt)
        
#         with st.chat_message("assistant"):
#         # response =[doc.content for doc in docs]
#             st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content":response})

#         history.chat_history.add_messages([
#             HumanMessage(content=prompt),
#             AIMessage(content=response)
#         ])
