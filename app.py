from config.settings import get_settings
from ui.streamlit import ChatbotUI
from services.chain import RAGChain
from services.history import ChatHistoryService
from services.llm import LLMService
from services.vector_store import VectorStoreService
from services.retriever import RetrieverService
from services.query_translation import QueryTranslationService

import langchain
langchain.verbose = False
langchain.tracing_enabled = False

def build_pipeline(path="./GermanCulture.pdf"):
    settings = get_settings()
    llm_service = LLMService(settings)
    vector_store_service = VectorStoreService(settings)
    chat_history_service = ChatHistoryService(settings)
    query_translation_service = QueryTranslationService(settings)
    retriever_service = RetrieverService(llm_service, vector_store_service, chat_history_service)
    return RAGChain(settings, llm_service, vector_store_service, retriever_service,
                    query_translation_service, chat_history_service), chat_history_service



if __name__ == "__main__":

    rag_chain, chat_history = build_pipeline()
    ui = ChatbotUI(rag_chain, chat_history)
    ui.run()
