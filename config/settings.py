import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

postgres_str = os.getenv("POSTGRES_CONNECTION_STR")
pgvector_str = os.getenv("PGVECTOR_CONNECTION_STR")
collection_name = os.getenv("COLLECTION_NAME")