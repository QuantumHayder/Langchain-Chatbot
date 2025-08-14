import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

postgres_str = os.getenv("POSTGRES_CONNECTION_STR")
pgvector_str = os.getenv("PGVECTOR_CONNECTION_STR")
collection_name = os.getenv("COLLECTION_NAME")

dbname = os.getenv("DBNAME")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")

embed_model = os.getenv("EMBED_MODEL_ID")
export_type = os.getenv("EXPORT_TYPE")