
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings
from services.document_loader import DocumentLoaderService

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

import psycopg

class VectorStoreService:
    """ This is a wrapper class"""
    def __init__(self, settings=None, embeddings=None):
        self.settings = settings or get_settings()
    
        # self.lc_embeddings = OpenAIEmbeddings(model= self.settings.embed_model_name,             
        # base_url=self.settings.base_url,       # e.g., "http://localhost:1234/v1"
        # api_key=self.settings.api_key,
        # check_embedding_ctx_length=False)
        
        #vector_store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings(check_embedding_ctx_length=False))
        self.li_embed_model = OpenAILikeEmbedding(
            model_name=self.settings.embed_model_id,
            api_base=self.settings.base_url,
            api_key="lm-studio",
            embed_batch_size=8,
            dimensions=768,
            check_embedding_ctx_length=False,        
        )
        
        # self.lc_vector_store = PGVector(
        # embeddings=self.lc_embeddings,
        # collection_name=self.settings.collection_name,
        # connection=self.settings.pgvector_str,
        # use_jsonb=True
        # )
        
        self.li_vector_store = PGVectorStore.from_params(
            database=self.settings.dbname,
            host=self.settings.host,
            password=self.settings.password,
            port=self.settings.port,
            user=self.settings.user,
            table_name="hierarchal_chunks",
            embed_dim=768,  # openai embedding dimension
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        self._doc_loader = DocumentLoaderService()
        
    # private helper method to check if the db is empty or not
    def _table_row_count(self) -> int:
        
        with psycopg.connect(
            dbname = self.settings.dbname,
            user=self.settings.user,
            password=self.settings.password,
            host=self.settings.host,
            port=self.settings.port
        ) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM data_hierarchal_chunks")
            return cur.fetchone()[0]
            
    def chunk_and_embed(self):
        storage_context = StorageContext.from_defaults(vector_store=self.li_vector_store)
        
        if(self._table_row_count() != 0):
            return VectorStoreIndex.from_vector_store(
                vector_store=self.li_vector_store,
                storage_context=storage_context,
                embed_model=self.li_embed_model,
            )
            
        nodes = self._doc_loader.llamaindex_file_loader(self.settings.path)
        return VectorStoreIndex(
            nodes, 
            storage_context=storage_context, 
            embed_model=self.li_embed_model
        )
        
        
# def chunk_and_embed(path): 
#     # Connect to your PostgreSQL database
#     conn = psycopg.connect(
#         dbname=settings.dbname,
#         user=settings.user,
#         password=settings.password,
#         host=settings.host,
#         port=settings.port
#     )
#     cursor = conn.cursor()

#     # Check if the table has any rows
#     cursor.execute("SELECT COUNT(*) FROM data_hierarchal_chunks")
#     row_count = cursor.fetchone()[0]

#     storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)

#     if row_count != 0:
#         index = VectorStoreIndex.from_vector_store(
#                 vector_store=llama_vector_store,
#                 storage_context=storage_context,
#                 embed_model=embed_model,
#             )
#         print("Table has data. Loaded index from existing vector store.")
#         return index


#     nodes = llamaindex_file_loader(path)
#     storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)
#     index = VectorStoreIndex(
#         nodes, storage_context=storage_context, embed_model=embed_model
#     )
#     print("Embedding and chunking completed.")

#     # Close the connection
#     cursor.close()
#     conn.close()
#     return index
