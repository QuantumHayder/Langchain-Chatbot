from services import history,chain
from langchain_core.messages import AIMessage, HumanMessage

def user_ask():
    user_input = input("Enter a question about german culture: ")
    
    history.add_user_message(user_input)
    
    docs = vector_store.similarity_search(user_input,k=2)
    for doc in docs:
        print(f"Page {doc.metadata['page']}: {doc.page_content[:300]}\n")
        
    response = chain.invoke(
        {
            "input": user_input
        }
    )
    
    history.add_ai_message(response.content)
    
    print(f"AI Response: {response.content}")
    print(f"Chat History: {history.messages}")
    
    
def get_response(user_query):
    user_chat_history = get_chat_history()
    result = chain.rag_chain.invoke({"input": user_query, "chat_history": user_chat_history})
    history.chat_history.add_message(HumanMessage(content=user_query))
    history.chat_history.add_message(AIMessage(content=result["answer"]))
    return result["answer"]

def get_chat_history():
    return history.chat_history.messages
