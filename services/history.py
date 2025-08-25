from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from config.settings import get_settings
import uuid
import psycopg

# session_id = str(uuid.uuid4())
# table_name:str="message_store"

# sync_connection = psycopg.connect(conninfo="dbname=mydb user=postgres password=BoodY2014 host=localhost port=5432")

# chat_history = PostgresChatMessageHistory (
#     table_name,
#     session_id ,
#     sync_connection = sync_connection
# )

class ChatHistoryService:
    """ This is a wrapper class to encapsulate the chat history functionality. """
    def __init__(self, settings=None, table_name:str = "message_store"):
        self.settings = settings or get_settings()
        self.table_name = table_name
        self.session_id = str(uuid.uuid4())
        self.sync_connection = psycopg.connect(
            conninfo=str(self.settings.postgres_str)
        )
        # underscore (_) is a convention in Python meaning “private” (not enforced)
        self._chat_history = PostgresChatMessageHistory(
            self.table_name,
            self.session_id,
            sync_connection=self.sync_connection
        )
        
    # @property turns a method into an attribute-like getter
    @property
    def chat_history(self):
        return self._chat_history
    
    def add_message(self, role, message):
        self._chat_history.add_message({"role": role, "content": message})
        
    def add_human_message(self, message):
        self._chat_history.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message):
        self._chat_history.add_message(AIMessage(content=message))
        
    def get_messages(self):
        return self._chat_history.messages
    
    def clear_history(self):
        self._chat_history.clear()
    