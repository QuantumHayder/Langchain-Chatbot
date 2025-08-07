from langchain_community.chat_message_histories import PostgresChatMessageHistory
from config import settings
import numpy as np

session_id = str(np.random.randint(1, 10000))

chat_history = PostgresChatMessageHistory (
    session_id = session_id,
    connection_string = settings.postgres_str
)