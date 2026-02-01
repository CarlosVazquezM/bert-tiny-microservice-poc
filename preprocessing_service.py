from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI(title="Preprocessing Service")

#Load tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

class TextRequest(BaseModel):
    text: str

class TokenizedResponse(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]

@app.post("/tokenize", response_model=TokenizedResponse)
def tokenize(request: TextRequest):
    tokens = tokenizer(
        request.text,      # The raw text to tokenize
        padding=True,      # Pad short sequences
        truncation=True,   # Cut long sequences
        max_length=128,    # Maximum tokens allowed
        return_tensors=None  # Output format
    )
    return TokenizedResponse(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"]
    )

@app.get("/health")
def health():
    return {"status":"healthy", "service":"preprocessing"}

