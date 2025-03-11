from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from ollama._types import ChatResponse
def allow_setattr(self, name, value):
    object.__setattr__(self, name, value)
ChatResponse.__setattr__ = allow_setattr
# Fim do monkey patch

app = Flask(__name__)

INDEX_DIR = "./index_storage"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = Ollama(model="jbrsolutions", request_timeout=3000.0, host="localhost")
Settings.llm = llm

storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, embed_model=embed_model)

query_engine = index.as_query_engine()

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Pergunta não fornecida"}), 400

    question_with_prompt = f"Responda em português (Brasil): {question}"
    response = query_engine.query(question_with_prompt)
    return jsonify({"answer": response.response, "metadata": response.metadata})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
