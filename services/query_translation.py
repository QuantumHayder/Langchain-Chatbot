from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from config.settings import get_settings
class QueryTranslationService:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        api_key = (
        self.settings.api_key.get_secret_value()
        if self.settings.api_key
        else "lm-studio"  # any non-empty string works for local servers
        )
        
        self.llm = ChatOpenAI(
            model=self.settings.model,
            base_url=self.settings.base_url,
            api_key=api_key,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "You produce exactly 2 alternative phrasings as a JSON array of strings." 
            "Return ONLY a valid JSON array of strings (no keys, no objects). No code fences, no extra text."
            ),
            ("human", "Original question: {question}")
        ])

        self._parser = JsonOutputParser()
        
        self._chain = self._prompt | self.llm | self._parser

    def generate_alternatives(self, question):
        return self._chain.invoke({"question": question})
    
    
# This returns a Python object parsed from JSON (ideally a list[str])


# template = """You are an AI language model assistant. Your task is to generate two 
# different versions of the given user question to retrieve relevant documents from a vector 
# database. By generating multiple perspectives on the user question, your goal is to help
# the user overcome some of the limitations of the distance-based similarity search. 
# Provide these alternative questions separated by newlines. Original question: {question}"""

# prompt_perspectives = ChatPromptTemplate.from_template(template)

# generate_queries = (
#     prompt_perspectives 
#     | ChatOpenAI(temperature=0) 
#     | StrOutputParser() 
#     | (lambda x: x.split("\n"))
# )

