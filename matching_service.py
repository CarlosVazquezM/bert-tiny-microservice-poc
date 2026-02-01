from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import httpx

app = FastAPI(title="Matching Service")

# Initialize ChromaDB (in-memory for POC)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="products")

PREPROCESSING_URL = "http://localhost:8001"
BERT_URL = "http://localhost:8002"

class SearchRequest(BaseModel):
    query : str
    top_k: int = 5

class AddItemRequest(BaseModel):
    id: str
    text: str

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]

async def get_embedding(text: str) -> list[float]:
    """Call preprocessing + BERT services to get embedding"""
    async with httpx.AsyncClient() as client:
        # Step 1: Tokenize
        token_response = await client.post(
            f"{PREPROCESSING_URL}/tokenize",
            json={"text": text}
        )
        tokens = token_response.json()
        
        # Step 2: Get embedding
        embed_response = await client.post(
            f"{BERT_URL}/embed",
            json=tokens
        )
        return embed_response.json()["embedding"]

@app.post("/add")
async def add_item(request: AddItemRequest):
    """Add an item to the vector store"""
    embedding = await get_embedding(request.text)
    collection.add(
        ids=[request.id],
        embeddings=[embedding],
        documents=[request.text]
    )
    return {"status": "added", "id": request.id}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for similar items"""
    query_embedding = await get_embedding(request.query)

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=request.top_k
    )

    search_results =[]
    for i, doc_id in enumerate(results["ids"][0]):
        search_results.append(SearchResult(
            id = doc_id,
            text = results["documents"][0][i],
            score=1- results["distances"][0][i] #Convert distance to similarity
        ))

    return SearchResponse(results=search_results)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "matching"}
