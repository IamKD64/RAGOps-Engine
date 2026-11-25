RAGOps-Engine

CI Show Image Show Image Show Image

A local-first Retrieval-Augmented Generation (RAG) system that answers questions over a technical document corpus with numbered source citations. It combines hybrid retrieval (dense vectors + BM25 fused with RRF), cross-encoder reranking, and a locally hosted LLM via Ollama, and ships with an evaluation harness so each retrieval strategy can be compared with real numbers.

No hosted LLM API or API key is required.

<!-- Add a screenshot or GIF of the UI here: ![RAGOps-Engine UI](docs/screenshot.png) -->
Features
Four retrieval modes, switchable per request: vector, bm25, hybrid, hybrid_rerank
Reciprocal Rank Fusion to merge dense and keyword rankings
Cross-encoder reranking of the fused candidate pool
Grounded answers: the model is instructed to answer only from numbered context passages, cite them as [1], [2], and say "I don't know" when the context lacks the answer
Per-request latency breakdown (retrieval vs. generation)
Evaluation harness: Recall@K and MRR per mode and difficulty tier, an optional CI recall gate, and an optional LLM-judge faithfulness check
React chat UI with a retrieval-mode selector, adjustable top-K, expandable source passages, and API status indicator
Docker, Pytest and GitHub Actions CI (unit tests, frontend build, Docker build)
Architecture
User question
FastAPI /ask
Vector searchChromaDB + bge-small
BM25 keyword search
Reciprocal Rank Fusion
Cross-encoder rerankerms-marco-MiniLM
Top-K context passages
Ollama LLMllama3.2:3b
Answer + citations +sources

In hybrid_rerank mode (the default), each of the two retrievers returns 20 candidates. RRF fuses them, the cross-encoder rescores the fused pool, and the top K (default 5) go into the prompt.

Tech stack
Layer	Technology
API	FastAPI, Uvicorn
Vector store	ChromaDB (persistent, cosine)
Embeddings	BAAI/bge-small-en-v1.5
Keyword search	rank-bm25 (BM25Okapi)
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2
LLM	llama3.2:3b via Ollama
Frontend	React 19, Vite
Tooling	Pytest, Docker, GitHub Actions
Corpus

29 documents from openly licensed documentation, fetched by scripts/build_corpus.py:

Source	Documents
Hugging Face Transformers docs	18
Hugging Face PEFT docs	4
A2A (Agent2Agent) protocol docs	7

Documents are cleaned (front matter, images and code blocks removed, links flattened) and split into 180-word chunks with 40-word overlap, producing 260 chunks. Chunk size and overlap are configurable via CHUNK_WORDS and CHUNK_OVERLAP.

Code blocks are stripped to keep the corpus prose-only, so questions about exact code syntax are out of scope.

Getting started
Prerequisites
Python 3.11
Ollama installed and running
Node.js 22+ (only for the frontend)
1. Install
bash
git clone https://github.com/<your-username>/RAGOps-Engine.git
cd RAGOps-Engine

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Configure
bash
cp .env.example .env               # Windows: Copy-Item .env.example .env
Variable	Default	Purpose
OLLAMA_URL	http://localhost:11434	Ollama server
LLM_MODEL	llama3.2:3b	Generation model
EMBED_MODEL	BAAI/bge-small-en-v1.5	Embedding model
RERANK_MODEL	cross-encoder/ms-marco-MiniLM-L-6-v2	Cross-encoder
CHUNK_WORDS	180	Words per chunk
CHUNK_OVERLAP	40	Overlap between chunks
3. Pull the LLM
bash
ollama pull llama3.2:3b
4. Build the index

The cleaned corpus is already in data/raw/:

bash
python -m app.ingest

To refresh or extend the corpus, edit SOURCES in scripts/build_corpus.py, run python scripts/build_corpus.py, then re-run ingestion.

5. Run the API
bash
uvicorn app.main:app --reload
API: http://127.0.0.1:8000
Interactive docs: http://127.0.0.1:8000/docs

The first request is slower because the embedding model and reranker load lazily.

6. Run the frontend (optional)
bash
cd frontend
npm ci
npm run dev

Vite proxies /api/* to http://127.0.0.1:8000, so start the API first.

Docker
bash
docker compose up --build

The container expects Ollama running on the host (host.docker.internal:11434) and mounts ./data for the index. Build the index first (step 4) so data/index/ exists. Embedding and reranker models are downloaded on first use.

API
GET /health
json
{ "status": "ok" }
POST /ask
Field	Type	Default	Notes
question	string	required	Minimum 3 characters
mode	string	hybrid_rerank	vector, bm25, hybrid, hybrid_rerank
k	integer	5	1 to 10 passages
bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does a client agent discover what a remote agent can do?", "mode": "hybrid_rerank", "k": 5}'

Response shape:

json
{
  "answer": "…grounded answer with citations like [1] and [2]…",
  "sources": [
    {
      "source": "A2A Protocol: Agent Discovery in A2A",
      "chunk_id": "a2a_agent_discovery-3",
      "text": "first 300 characters of the passage…"
    }
  ],
  "latency_ms": { "retrieval": 120, "generation": 4300 }
}

Errors: 400 for an invalid mode, 503 if the Ollama backend is unreachable.

Evaluation
bash
python -m eval.run_eval                     # compare all modes
python -m eval.run_eval --k 5 --min-recall 0.9   # fail (exit 1) if hybrid_rerank recall is below threshold
python -m eval.run_eval --judge 10          # also run the LLM faithfulness check (needs Ollama)

The golden set (eval/golden.jsonl) has 41 questions: 29 easy and 12 hard. A retrieval counts as a hit when the expected source document appears in the top K. Results are written to eval/results.json.

Recall@5 / MRR, all 41 questions:

Mode	Recall@5	MRR
vector	0.951	0.850
bm25	0.902	0.764
hybrid	0.951	0.837
hybrid_rerank	1.000	0.888

Hard tier only (12 questions):

Mode	Recall@5	MRR
vector	0.917	0.694
bm25	0.750	0.569
hybrid	0.833	0.632
hybrid_rerank	1.000	0.757

How to read these numbers

Reranking gives the clearest gain, especially on the hard tier.
Plain hybrid does not beat vector on the hard tier, so fusion alone isn't a guaranteed win.
The corpus is small (29 documents), matching is at the document level rather than the passage level, and the hard tier has only 12 questions. Treat these as a comparison between strategies on this corpus, not as a general accuracy claim.
The LLM-judge faithfulness check is optional and not included in the committed results.
Project structure
text
RAGOps-Engine/
├── app/
│   ├── main.py          # FastAPI app (/health, /ask)
│   ├── retrieval.py     # Retriever: vector, BM25, hybrid, rerank
│   ├── fusion.py        # Reciprocal Rank Fusion
│   ├── chunking.py      # Word-window chunking with overlap
│   ├── ingest.py        # Builds ChromaDB index + chunks.json
│   ├── llm.py           # Prompt construction + Ollama client
│   └── config.py        # Env-driven settings
├── data/
│   ├── raw/             # Cleaned source documents (29)
│   └── index/           # Generated index (gitignored)
├── eval/
│   ├── golden.jsonl     # Evaluation questions
│   ├── run_eval.py      # Recall@K, MRR, faithfulness judge
│   └── results.json     # Latest results
├── frontend/            # React + Vite chat UI
├── scripts/
│   └── build_corpus.py  # Downloads and cleans docs
├── tests/
│   └── test_units.py    # Chunking and RRF unit tests
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
Testing and CI
bash
python -m pytest -q tests

GitHub Actions runs unit tests, a production frontend build, and a Docker image build on every push and pull request to main.

Limitations
Small, single-domain corpus; the evaluation set is small and document-level.
A 3B-parameter local model can miss citations or over-answer; answer quality depends heavily on the model you choose.
Code blocks are removed from the corpus.
No authentication, rate limiting, or streaming; intended for local use and demos.
Roadmap
[ ] Passage-level evaluation and a larger golden set, including unanswerable questions
[ ] Run the recall gate in CI
[ ] API tests with a mocked retriever and LLM
[ ] Streaming responses
[ ] Heading-aware chunking with document titles prepended
Acknowledgements

Documentation corpus from the Hugging Face Transformers, PEFT and A2A repositories (Apache-2.0 licensed).

License

Add a LICENSE file and state it here (MIT or Apache-2.0 are common choices).
