from services import retriever,llm

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm.llm, llm.qa_prompt)
rag_chain = create_retrieval_chain(
    retriever.history_aware_retriever, question_answer_chain
)
