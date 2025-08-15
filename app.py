#from langchain_core.vectorstores import InMemoryVectorStore
from ui.streamlit import chatbot_ask
from services.vector_store import llama_vector_store

from config import settings


if __name__ == "__main__":

    chatbot_ask()