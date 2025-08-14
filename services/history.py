from langchain_postgres import PostgresChatMessageHistory
from config import settings
import uuid
import psycopg

session_id = str(uuid.uuid4())
table_name:str="message_store"

sync_connection = psycopg.connect(conninfo="dbname=mydb user=postgres password=BoodY2014 host=localhost port=5432")

chat_history = PostgresChatMessageHistory (
    table_name,
    session_id ,
    sync_connection = sync_connection
)