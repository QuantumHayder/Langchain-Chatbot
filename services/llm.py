from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import settings

llm = ChatOpenAI(
    model = settings.model, 
    base_url = settings.base_url,
    openai_api_key = settings.api_key,
    
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable and efficient AI assistant. Your role is to search for relevant information from the in-memory vector store and provide accurate, concise answers to user questions based on the retrieved content. Do not generate information beyond what is found in the vector store unless explicitly asked to elaborate. /no_think",
        ),
        ("human","{input}"),
    ]
)

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Answer question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


#chain =  prompt | llm 
