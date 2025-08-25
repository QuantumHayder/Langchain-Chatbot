from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_history_aware_retriever
            
from typing import Any, List

from services.llm import LLMService
from services.vector_store import VectorStoreService
from services.history import ChatHistoryService
class RetrieverService:
    
    def __init__(self, llm: LLMService, vector_store: VectorStoreService, chat_history: ChatHistoryService):
        self.llm = llm
        self.vector_store = vector_store
        self.chat_history = chat_history
        
        self._chat_doc_retriever: RetrieverLike = RunnableLambda(
            self._chat_to_documents
        )
        
        self._history_aware_retriever: RetrieverLike = create_history_aware_retriever(
            self.llm.llm,
            self._chat_doc_retriever,
            self.llm.contextualize_q_prompt
        )
        
    def _chat_to_documents(self, _: Any) -> List[Document]:
        """Convert chat history to documents."""
        msgs = self.chat_history.get_messages()[-10:]
        docs: List[Document] = []
        for msg in msgs:
            content = getattr(msg, "content", str(msg))
            docs.append(
                Document(
                    page_content=content,
                    metadata={"message_type": msg.__class__.__name__},
                )
            )
        return docs
    
    @property
    def history_aware_retriever(self):
        return self._history_aware_retriever
    
    def retrieve_answer(self, query: str) -> str:
        """Retrieve an answer based on the query."""
        retriever = self.vector_store.lc_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        chain = RetrievalQA.from_chain_type(llm=self.llm.llm, chain_type="stuff", retriever=retriever)
        answer = chain.invoke({"query": query})
        return answer

# def retrieve_answer(query):

#     # similarity_with_score
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever = retriever)
#     answer = chain.invoke({"query": query})
#     return answer
# ########################################################

# retriever: RetrieverLike = RunnableLambda(
#     lambda query: [
#         Document(
#             page_content=msg.content,
#             metadata={"message_type": msg.__class__.__name__}
#         )
#         for msg in chat_history.messages[-10:]  # Last 10 messages
#     ]
# )

# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )