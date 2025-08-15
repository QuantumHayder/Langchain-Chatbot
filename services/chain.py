from services import retriever, llm, vector_store

from langchain.retrievers import EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import LlamaIndexRetriever as LCLlamaIndexRetriever
    
from llama_index.core.retrievers import VectorIndexRetriever

vector_index = vector_store.chunk_and_embed("./GermanCulture.pdf")
li_vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

try:
    lc_retriever = li_vector_retriever.as_langchain()  # available in recent LlamaIndex versions
except AttributeError:
    # Fallback: use the LangChain wrapper around LlamaIndex
    lc_retriever = LCLlamaIndexRetriever(index=vector_index)  # pass index; tune kwargs as needed

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever.history_aware_retriever, lc_retriever],
    weights=[0.4, 0.6]
)

question_answer_chain = create_stuff_documents_chain(llm.llm, llm.qa_prompt)
rag_chain = create_retrieval_chain(
    ensemble_retriever, question_answer_chain
)
