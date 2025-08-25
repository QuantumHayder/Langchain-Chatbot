# import os
# from dotenv import load_dotenv
from functools import lru_cache

from typing import Optional

from pydantic import (
    SecretStr,
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    PostgresDsn,
    AnyUrl,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    # LLM Configuration
    api_key: Optional[SecretStr] = Field(default=None, env='OPENAI_API_KEY')
    base_url: str = Field(alias='OPENAI_BASE_URL')
    model: str = Field(alias='OPENAI_MODEL')
    
    # Database Configuration
    postgres_str: PostgresDsn = Field(alias='POSTGRES_CONNECTION_STR')
    pgvector_str: PostgresDsn = Field(alias='PGVECTOR_CONNECTION_STR')
    collection_name: str = Field(alias='COLLECTION_NAME')
    
    # PostgreSQL Connection
    dbname: str = Field(alias='DBNAME')
    user: str = Field(alias='USER')
    password: str = Field(alias='PASSWORD')
    host: str = Field(alias='HOST')
    port: int = Field(alias='PORT')
    
    # Embedding Model
    embed_model_id: str = Field(alias='EMBED_MODEL_ID')
    embed_model_name: str = Field(alias='EMBED_MODEL_NAME')
    
    # Export Type
    export_type: str = Field(alias='EXPORT_TYPE')
    
    # Pydantic settings
    model_config = SettingsConfigDict(
        env_file=".env",              # load from .env automatically
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,                  # make it immutable
        case_sensitive=False,
    )
    
    # File Path
    path: str = Field(alias='FILE_PATH', default='./GermanCulture.pdf')
    
@lru_cache(maxsize=128)
def get_settings() -> Settings:
    return Settings()



# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# base_url = os.getenv("OPENAI_BASE_URL")
# model = os.getenv("OPENAI_MODEL")

# postgres_str = os.getenv("POSTGRES_CONNECTION_STR")
# pgvector_str = os.getenv("PGVECTOR_CONNECTION_STR")
# collection_name = os.getenv("COLLECTION_NAME")

# dbname = os.getenv("DBNAME")
# user = os.getenv("USER")
# password = os.getenv("PASSWORD")
# host = os.getenv("HOST")
# port = os.getenv("PORT")

# embed_model = os.getenv("EMBED_MODEL_ID")
# export_type = os.getenv("EXPORT_TYPE")