import os
import shutil
import pandas as pd
from sqlalchemy import create_engine
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PG_HOST = os.getenv("PG_HOST", "localhost")
DATABASE = os.getenv("DATABASE", "mydb")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "123")
PG_PORT = os.getenv("PG_PORT", "5432")

engine = create_engine(f'postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{DATABASE}')

query = "SELECT id, texto, sensivel FROM conhecimento"
df = pd.read_sql(query, engine)

documents = [
    Document(text=row["texto"], metadata={"id": row["id"], "sensivel": row["sensivel"]})
    for _, row in df.iterrows()
]

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = "./index_storage"

if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

if os.path.exists(INDEX_DIR):
    shutil.rmtree(INDEX_DIR)

os.makedirs(INDEX_DIR)

index = VectorStoreIndex(documents, embed_model=embed_model)
index.storage_context.persist(persist_dir=INDEX_DIR)

print("Treinamento finalizado!")