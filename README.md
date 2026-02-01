# BERT-Tiny Semantic Search Microservices POC

A proof-of-concept demonstrating a microservices architecture for semantic search using BERT-Tiny embeddings. This project implements the pattern used by companies for real-time product search.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Preprocessing     │     │     BERT-Tiny       │     │      Matching       │
│     Service         │ ──▶ │      Service        │ ──▶ │      Service        │
│                     │     │                     │     │                     │
│  - Tokenizes text   │     │  - Receives tokens  │     │  - Stores embeddings│
│  - Returns token    │     │  - Generates        │     │  - Vector similarity│
│    IDs + mask       │     │    embeddings       │     │    search           │
│                     │     │                     │     │                     │
│  Port: 8001         │     │  Port: 8002         │     │  Port: 8003         │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
                                                                  ▼
                                                        ┌─────────────────────┐
                                                        │      ChromaDB       │
                                                        │   (Vector Store)    │
                                                        └─────────────────────┘
```

## Why This Architecture?

**Separation of Concerns**: Each service has a single responsibility, making them easier to maintain, test, and scale independently.

**Independent Scaling**: 
- High traffic? Scale the matching service
- GPU-intensive workload? Scale BERT service with beefier instances
- Tokenization is CPU-light and needs minimal resources

**Technology Flexibility**: Swap BERT-Tiny for a larger model without changing other services.

## How It Works

1. **Preprocessing Service**: Tokenizes raw text into token IDs using BERT's WordPiece tokenizer
2. **BERT-Tiny Service**: Converts token IDs into 128-dimensional embeddings using mean pooling
3. **Matching Service**: Stores embeddings in ChromaDB and performs similarity search

### Key Concepts

- **BERT-Tiny**: A distilled version of BERT with only 4.4M parameters (vs 110M for BERT-Base), optimized for fast inference
- **Embeddings**: Dense vector representations that capture semantic meaning
- **Vector Search**: Finding similar items by comparing embedding distances rather than keyword matching

## Prerequisites

- Python 3.8–3.12 (3.11 recommended)
- pip

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CarlosVazquezM/bert-tiny-microservice-poc.git
   cd bert-tiny-microservices-poc
   ```

2. **Create and activate virtual environment**
   

If you have different Python version in your computer use pyenv 
to create it with 3.11 > Recommended
Run it before creating your virtual environment.
   ```bash
   #pyenv local 3.11

   python -m venv .venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install transformers torch fastapi uvicorn chromadb httpx
   ```

   If you encounter issues with PyTorch, install it separately first:
   ```bash
   # CPU-only (recommended for this POC)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

## Running the Services

Open three separate terminal windows, activate the virtual environment in each, and run:

**Terminal 1 - Preprocessing Service:**
```bash
uvicorn preprocessing_service:app --port 8001 --reload
```

**Terminal 2 - BERT Service:**
```bash
uvicorn bert_service:app --port 8002 --reload
```

**Terminal 3 - Matching Service:**
```bash
uvicorn matching_service:app --port 8003 --reload
```

## Usage

### Add Items

```bash
curl -X POST http://localhost:8003/add \
  -H "Content-Type: application/json" \
  -d '{"id": "1", "text": "red running sneakers for men"}'

curl -X POST http://localhost:8003/add \
  -H "Content-Type: application/json" \
  -d '{"id": "2", "text": "blue leather dress shoes"}'

curl -X POST http://localhost:8003/add \
  -H "Content-Type: application/json" \
  -d '{"id": "3", "text": "comfortable walking shoes red color"}'

curl -X POST http://localhost:8003/add \
  -H "Content-Type: application/json" \
  -d '{"id": "4", "text": "black formal oxford shoes"}'
```

### Search

```bash
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red sneakers", "top_k": 3}'
```

**Expected Result**: Items 1 and 3 should rank highest due to semantic similarity, even though item 3 doesn't contain the word "sneakers".

### Health Checks

```bash
curl http://localhost:8001/health  # Preprocessing
curl http://localhost:8002/health  # BERT-Tiny
curl http://localhost:8003/health  # Matching
```

## API Reference

### Preprocessing Service (Port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tokenize` | POST | Tokenize text into token IDs |
| `/health` | GET | Health check |

**Request Body:**
```json
{
  "text": "your text here"
}
```

**Response:**
```json
{
  "input_ids": [101, 2417, 20112, 102],
  "attention_mask": [1, 1, 1, 1]
}
```

### BERT-Tiny Service (Port 8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate embeddings from tokens |
| `/health` | GET | Health check |

**Request Body:**
```json
{
  "input_ids": [101, 2417, 20112, 102],
  "attention_mask": [1, 1, 1, 1]
}
```

**Response:**
```json
{
  "embedding": [0.23, 0.51, -0.12, ...]
}
```

### Matching Service (Port 8003)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add` | POST | Add item to vector store |
| `/search` | POST | Search for similar items |
| `/health` | GET | Health check |

## Project Structure

```
bert-tiny-microservices-poc/
├── preprocessing_service.py   # Tokenization service
├── bert_service.py            # BERT-Tiny embedding service
├── matching_service.py        # Vector search service
├── requirements.txt           # Python dependencies
└── README.md
```

## Technical Details

### BERT-Tiny Model
- **Model**: `prajjwal1/bert-tiny` from Hugging Face
- **Parameters**: 4.4M (vs 110M for BERT-Base)
- **Hidden Size**: 128 dimensions
- **Layers**: 2
- **Inference**: ~5ms per query

### Embedding Strategy
This POC uses **mean pooling** - averaging all token embeddings to create a single sentence embedding. Alternative strategies include:
- CLS token pooling (using only the [CLS] token embedding)
- Max pooling (taking maximum values across tokens)

### Vector Storage
ChromaDB is used as an in-memory vector store for simplicity. For production, consider:
- **Pinecone**: Fully managed, scalable
- **Weaviate**: Self-hosted with hybrid search
- **Milvus**: Large-scale open source
- **pgvector**: PostgreSQL extension

## Extending This POC

### Ideas for Enhancement
1. **Add Docker**: Containerize each service
2. **Add Kubernetes**: Deploy with k8s for orchestration
3. **Swap Vector DB**: Replace ChromaDB with Pinecone for persistence
4. **Add Caching**: Redis layer for frequently searched queries
5. **Batch Processing**: Support bulk embedding generation
6. **Metrics**: Add Prometheus metrics for monitoring

### Production Considerations
- Pre-compute product embeddings offline (batch job)
- Use GPU instances for BERT service at scale
- Add authentication and rate limiting
- Implement circuit breakers between services
- Add comprehensive logging and tracing

## License

MIT

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library and model hosting
- [ChromaDB](https://www.trychroma.com/) for the vector database
- Architecture pattern inspired by real-world implementations at e-commerce retailers.