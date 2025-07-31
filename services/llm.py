from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
            "You are a helpful ai assistant that searches for data in the in-memory vector store and answers the user questions .",
        ),
        ("human","{input}"),
    ]
)

chain =  prompt | llm 
