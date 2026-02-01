from fastapi import FastAPI
from pydantic import BaseModel
import torch 
from transformers import AutoModel

app = FastAPI(title="BERT-Tiny Service")

#Load model once at startup
# use safetensors to load models (avoids torch 2.6 requirement)
model = AutoModel.from_pretrained("prajjwal1/bert-tiny", use_safetensors=True)
model.eval()

class TokenizedInput(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@app.post("/embed", response_model=EmbeddingResponse)
def embed(request: TokenizedInput):
    with torch.no_grad():
        inputs = {
            "input_ids": torch.tensor([request.input_ids]),
            "attention_mask": torch.tensor([request.attention_mask])
        }
        outputs = model(**inputs)

        #Mean pooling: average all token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return EmbeddingResponse(embedding=embedding)

@app.get("/health")
def health():
    return {"status":"healthy","service":"bert-tiny"}
        
