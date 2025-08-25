from services import retriever, llm, vector_store, query_translation
from utils import helper


from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    
    def __init__(self, settings, llm, vector_store, retriever, query_translation, chat_history):
        self.settings = settings
        self.llm = llm
        self.vector_store = vector_store
        self.retriever = retriever
        self.query_translation = query_translation
        self.chat_history = chat_history
        
        self._index = self.vector_store.chunk_and_embed()
        self._query_engine = self._index.as_query_engine(llm=self.llm.llm)

    def _extract_texts_from_response(self, resp) -> List[str]:
        """Extract source chunk text from a response."""

        texts: List[str] = []
        try:
            if hasattr(resp, "source_nodes") and resp.source_nodes:
                for sn in resp.source_nodes:
                    if hasattr(sn, "get_text") and callable(sn.get_text):
                        texts.append(sn.get_text())
                    elif hasattr(sn, "node") and hasattr(sn.node, "get_text"):
                        texts.append(sn.node.get_text())
                    elif hasattr(sn, "text"):
                        texts.append(sn.text)
        except Exception:
            pass
        if not texts:
            texts.append(str(resp))
        return texts

    
    def invoke(self, user_query: str) -> str:
        # 1) Generate alternative phrasings
        questions = self.query_translation.generate_alternatives(user_query)
        if not isinstance(questions, list):
            questions = [str(questions)]
        questions = [str(q).strip() for q in questions if str(q).strip()]

        # 2) Retrieve from LlamaIndex (vector index)
        retrieved_texts: List[str] = []
        for q in questions:
            resp = self._query_engine.query(q)
            retrieved_texts.extend(self._extract_texts_from_response(resp))

        chat_docs = self.retriever.history_aware_retriever.invoke(
            {
                "input": user_query,
                "chat_history": self.chat_history.get_messages(),
            }
        )
        chat_texts = [getattr(cd, "page_content", str(cd)) for cd in chat_docs]

        # 4) Merge contexts and run QA
        all_texts = [t for t in (retrieved_texts + chat_texts) if isinstance(t, str) and t.strip()]
        context_str = "\n\n".join(all_texts)

        qa_chain = self.llm.qa_prompt | self.llm.llm | StrOutputParser()
        return qa_chain.invoke({"input": user_query, "context": context_str})

    
    
# vector_index = vector_store.chunk_and_embed("./GermanCulture.pdf")
# query_engine = vector_index.as_query_engine(llm=llm.llm)

# chat_retriever = retriever.history_aware_retriever


# def _extract_texts_from_llamaindex_response(resp) -> List[str]:
#     """Try to extract source chunk text from a LlamaIndex Response across versions."""
#     texts = []
#     try:
#         # Newer LlamaIndex: resp.source_nodes is often populated
#         if hasattr(resp, "source_nodes") and resp.source_nodes:
#             for sn in resp.source_nodes:
#                 # Common access patterns across versions
#                 if hasattr(sn, "get_text") and callable(sn.get_text):
#                     texts.append(sn.get_text())
#                 elif hasattr(sn, "node") and hasattr(sn.node, "get_text"):
#                     texts.append(sn.node.get_text())
#                 elif hasattr(sn, "text"):
#                     texts.append(sn.text)
#     except Exception:
#         pass
#     # Fallback to the synthesized response text if no chunks were extracted
#     if not texts:
#         texts.append(str(resp))
#     return texts


# def invocation_series(user_query):
#     questions = query_translation.generate_queries.invoke({"question": user_query})
    
#     if isinstance(questions, dict) and "questions" in questions:
#         questions = questions["questions"]
#     elif isinstance(questions, str):
#         questions = [q.strip() for q in questions.split("\n") if q.strip()]
#     elif isinstance(questions, list):
#         questions = [str(q).strip() for q in questions if str(q).strip()]
#     else:
#         questions = [str(questions).strip()]

#     # Ensure list[str]
#     # questions = [str(q).strip() for q in questions if str(q).strip()]

#     retrieved_texts = []
#     for q in questions:
#         print(q, ': ',"\n")
#         resp = query_engine.query(q)
#         print(resp, ': ',"\n")
#         retrieved_texts.extend(_extract_texts_from_llamaindex_response(resp))
        
#     chat_docs = chat_retriever.invoke({"input": user_query, "chat_history":helper.get_chat_history()})
#     chat_text = []
#     for cd in chat_docs:
#         if hasattr(cd, "page_content"):
#             chat_text.append(cd.page_content)
#         else:
#             # fallback if not a Document
#             chat_text.append(str(cd))

#     all_texts = [t for t in (retrieved_texts + chat_text) if isinstance(t, str) and t.strip()]
#     context_str = "\n\n".join(all_texts)

#     qa_chain = llm.qa_prompt | llm.llm | StrOutputParser()
#     result = qa_chain.invoke({"input": user_query, "context": context_str})

#     return result

# """ search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7}"""