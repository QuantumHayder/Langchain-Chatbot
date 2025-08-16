from services import retriever, llm, vector_store, query_translation
from utils import helper


from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser


vector_index = vector_store.chunk_and_embed("./GermanCulture.pdf")
query_engine = vector_index.as_query_engine(llm=llm.llm)

chat_retriever = retriever.history_aware_retriever


def _extract_texts_from_llamaindex_response(resp) -> List[str]:
    """Try to extract source chunk text from a LlamaIndex Response across versions."""
    texts = []
    try:
        # Newer LlamaIndex: resp.source_nodes is often populated
        if hasattr(resp, "source_nodes") and resp.source_nodes:
            for sn in resp.source_nodes:
                # Common access patterns across versions
                if hasattr(sn, "get_text") and callable(sn.get_text):
                    texts.append(sn.get_text())
                elif hasattr(sn, "node") and hasattr(sn.node, "get_text"):
                    texts.append(sn.node.get_text())
                elif hasattr(sn, "text"):
                    texts.append(sn.text)
    except Exception:
        pass
    # Fallback to the synthesized response text if no chunks were extracted
    if not texts:
        texts.append(str(resp))
    return texts


def invocation_series(user_query):
    questions = query_translation.generate_queries.invoke({"question": user_query})
    
    if isinstance(questions, dict) and "questions" in questions:
        questions = questions["questions"]
    elif isinstance(questions, str):
        questions = [q.strip() for q in questions.split("\n") if q.strip()]
    elif isinstance(questions, list):
        questions = [str(q).strip() for q in questions if str(q).strip()]
    else:
        questions = [str(questions).strip()]

    # Ensure list[str]
    # questions = [str(q).strip() for q in questions if str(q).strip()]

    retrieved_texts = []
    for q in questions:
        print(q, ': ',"\n")
        resp = query_engine.query(q)
        print(resp, ': ',"\n")
        retrieved_texts.extend(_extract_texts_from_llamaindex_response(resp))
        
    chat_docs = chat_retriever.invoke({"input": user_query, "chat_history":helper.get_chat_history()})
    chat_text = []
    for cd in chat_docs:
        if hasattr(cd, "page_content"):
            chat_text.append(cd.page_content)
        else:
            # fallback if not a Document
            chat_text.append(str(cd))

    all_texts = [t for t in (retrieved_texts + chat_text) if isinstance(t, str) and t.strip()]
    context_str = "\n\n".join(all_texts)

    qa_chain = llm.qa_prompt | llm.llm | StrOutputParser()
    result = qa_chain.invoke({"input": user_query, "context": context_str})

    return result

""" search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7}"""