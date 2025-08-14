#from langchain_core.vectorstores import InMemoryVectorStore
from ui.streamlit import chatbot_ask
from services.vector_store import llama_vector_store

from config import settings


if __name__ == "__main__":
   
    index = chunk_and_embed("testpdf.pdf")
    retriever = index.as_retriever()

    chatbot_ask()