
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

from config import settings
from services.document_loader import llamaindex_file_loader

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5@q8_0",check_embedding_ctx_length=False)
#vector_store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings(check_embedding_ctx_length=False))

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=settings.collection_name,
    connection=settings.pgvector_str,
    use_jsonb=True
)

#################################################################################

llama_vector_store = PGVectorStore.from_params(
    database=settings.dbname,
    host=settings.host,
    password=settings.password,
    port=settings.port,
    user=settings.user,
    table_name="hierarchal_chunks",
    embed_dim=768,  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

embed_model = OpenAILikeEmbedding(
    model_name=settings.embed_model,
    api_base=settings.base_url,
    api_key="lm-studio",
    embed_batch_size=8,
    dimensions=768,
    check_embedding_ctx_length=False,        
)

def chunk_and_embed(path): 
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        host=settings.host,
        port=settings.port
    )
    cursor = conn.cursor()

    # Check if the table has any rows
    cursor.execute("SELECT COUNT(*) FROM hierarchal_chunks")
    row_count = cursor.fetchone()[0]

    if row_count != 0:
        print("Table already has data. Skipping embedding and chunking.")
        return

    nodes = llamaindex_file_loader(path)
    storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)
    index = VectorStoreIndex(
        nodes, storage_context=storage_context, embed_model=embed_model
    )
    print("Embedding and chunking completed.")

    # Close the connection
    cursor.close()
    conn.close()
    return index



