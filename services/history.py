from langchain_community.chat_message_histories import PostgresChatMessageHistory
from config import settings
import numpy as np

session_id = np.random.randint(1, 10000)

history = PostgresChatMessageHistory (
    session_id = session_id,
    connection_string = settings.postgres_str
)