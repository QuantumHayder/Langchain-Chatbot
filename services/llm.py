from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.settings import get_settings

# Default Templates
class LLMService:
    """ This is a wrapper class to encapsulate the LLM functionality. """
    
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable and efficient AI assistant."
            "Your role is to search for relevant information from the in-memory vector store" 
            "and provide accurate, concise answers to user questions based on the retrieved content."
            "Do not generate information beyond what is found in the vector store unless explicitly asked to elaborate."
            "\n\n"
            "/no_think",
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
        "\n\n"
        "/no_think"
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
        "/no_think"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            # MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    def __init__(self,settings=None):
        self.settings = settings or get_settings()
        
        
        api_key = (
        self.settings.api_key.get_secret_value()
        if self.settings.api_key
        else "lm-studio"  # any non-empty string works for local servers
        )

        self.llm = ChatOpenAI(
            model = self.settings.model, 
            base_url = self.settings.base_url,
            api_key = api_key,
        )
