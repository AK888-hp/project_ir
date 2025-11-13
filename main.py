from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

app=FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name = "medical_docs")

class EmbedRequest(BaseModel):
    text:str

class IndexRequest(BaseModel):
    id: str
    vector: list
    document: str

@app.get("/")
async def read_root():
    return {"message":"AI Service is alive and running!"}

@app.post("/embed")
async def create_embedding(request_data: EmbedRequest):
    text_to_embed = request_data.text
    embedding=model.encode(text_to_embed)
    embedding_as_list=embedding.tolist()
    return {"embedding":embedding_as_list}

@app.post("/add_to_index")
async def add_of_index(request_data:IndexRequest):
    collection.add(embeddings=[request_data.vector],documents=[request_data.document],ids=[request_data.id])
    return {"status":"success","id_added":request_data.id}
    
@app.post("/query")
async def query_index(request_data:EmbedRequest):
    query_text = request_data.text
    query_embedding = model.encode(query_text)
    query_vector_as_list = query_embedding.tolist()
    results = collection.query(query_embeddings=[query_vector_as_list],n_results=2)
    return results
