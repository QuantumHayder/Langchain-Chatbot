
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from config import settings

embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5@q8_0",check_embedding_ctx_length=False)
#vector_store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings(check_embedding_ctx_length=False))

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=settings.collection_name,
    connection=settings.pgvector_str,
    use_jsonb=True
)

