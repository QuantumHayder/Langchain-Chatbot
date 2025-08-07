#from langchain_core.vectorstores import InMemoryVectorStore
from ui.streamlit import chatbot_ask
from services.vector_store import vector_store
from services.document_loader import load_and_split_pdf


if __name__ == "__main__":
    docs = load_and_split_pdf("GermanCulture.pdf")
    vector_store.add_documents(docs)  # Add documents to the vector store
    chatbot_ask()