from services.history import chat_history
from services.vector_store import vector_store
from services.llm import llm, contextualize_q_prompt

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_history_aware_retriever

def retrieve_answer(query):
   
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever = retriever)
    answer = chain.invoke({"query": query})
    return answer

retriever: RetrieverLike = RunnableLambda(
    lambda query: [
        Document(
            page_content=msg.content,
            metadata={"message_type": msg.__class__.__name__}
        )
        for msg in chat_history.messages[-10:]  # Last 10 messages
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)