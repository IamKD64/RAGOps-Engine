\# Production RAG Assistant



A Retrieval-Augmented Generation (RAG) application built with \*\*FastAPI, ChromaDB, BM25, Sentence Transformers, Cross-Encoder reranking, and Ollama\*\*.



The system retrieves relevant passages from a curated collection of AI-industry documents and uses a locally running language model to generate grounded answers with source citations.



\## Features



\- \*\*Hybrid retrieval:\*\* Combines semantic vector search and BM25 keyword search.

\- \*\*Reciprocal Rank Fusion (RRF):\*\* Combines results from different retrieval methods.

\- \*\*Cross-Encoder reranking:\*\* Reorders retrieved candidates by relevance.

\- \*\*Local LLM inference:\*\* Uses Ollama, avoiding dependence on hosted LLM APIs.

\- \*\*Grounded generation:\*\* Answers using retrieved context and provides numbered citations.

\- \*\*Evaluation framework:\*\* Compares retrieval strategies using Recall@K and Mean Reciprocal Rank (MRR).

\- \*\*Faithfulness evaluation:\*\* Optional LLM-based checking of whether generated claims are supported by retrieved context.

\- \*\*REST API:\*\* Exposes the RAG pipeline through FastAPI.



\## Architecture



```text

&#x20;                User Question

&#x20;                      |

&#x20;                      v

&#x20;                FastAPI API

&#x20;                      |

&#x20;                      v

&#x20;                Query Retrieval

&#x20;                      |

&#x20;            +---------+---------+

&#x20;            |                   |

&#x20;            v                   v

&#x20;       Vector Search          BM25 Search

&#x20;       (ChromaDB)             (Keyword)

&#x20;            |                   |

&#x20;            +---------+---------+

&#x20;                      |

&#x20;                      v

&#x20;             Reciprocal Rank Fusion

&#x20;                      |

&#x20;                      v

&#x20;              Cross-Encoder

&#x20;                Reranking

&#x20;                      |

&#x20;                      v

&#x20;               Top-K Context

&#x20;                      |

&#x20;                      v

&#x20;                Ollama LLM

&#x20;                      |

&#x20;                      v

&#x20;            Grounded Answer

&#x20;             + Citations

```



\## Technology Stack



| Component | Technology |

|---|---|

| Programming language | Python 3.11 |

| API framework | FastAPI |

| Vector database | ChromaDB |

| Embeddings | BAAI/bge-small-en-v1.5 |

| Keyword retrieval | BM25 |

| Result fusion | Reciprocal Rank Fusion |

| Reranking | Cross-Encoder |

| Language model | Llama 3.2 3B |

| Local inference | Ollama |

| Testing | Pytest |

| Evaluation | Recall@K, MRR, LLM judge |



\## Project Structure



```text

project1-rag/

│

├── app/

│   ├── chunking.py

│   ├── config.py

│   ├── fusion.py

│   ├── ingest.py

│   ├── llm.py

│   ├── main.py

│   └── retrieval.py

│

├── data/

│   ├── raw/                 # Original document corpus

│   └── index/               # Generated vector index

│

├── eval/

│   ├── golden.jsonl         # Evaluation questions

│   ├── results.json         # Evaluation results

│   └── run\_eval.py

│

├── scripts/

│   └── build\_corpus.py

│

├── tests/

│   └── test\_units.py

│

├── .env.example

├── .gitignore

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

```



\## Document Collection



The project uses a curated collection of \*\*29 AI-industry documents\*\* covering:



\- Hugging Face Transformers

\- Parameter-Efficient Fine-Tuning (PEFT)

\- Agent-to-Agent (A2A) protocol



The document collection is divided into overlapping text chunks for retrieval.



Default chunking configuration:



\- Chunk size: 180 words

\- Chunk overlap: 40 words



\## Getting Started



\### 1. Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd project1-rag

```



\### 2. Create a virtual environment



Windows PowerShell:



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

pip install -r requirements.txt

```



\### 4. Configure environment variables



```powershell

Copy-Item .env.example .env

```



Default configuration:



```env

OLLAMA\_URL=http://localhost:11434

LLM\_MODEL=llama3.2:3b

EMBED\_MODEL=BAAI/bge-small-en-v1.5

RERANK\_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

```



\### 5. Install and start Ollama



Install Ollama from:



https://ollama.com/



Download the model:



```powershell

ollama pull llama3.2:3b

```



Make sure Ollama is running before sending API requests.



\### 6. Build the vector index



If the original documents are already present in `data/raw/`, run:



```powershell

python -m app.ingest

```



The ingestion pipeline chunks the documents, generates embeddings, and builds the ChromaDB index.



If the raw corpus is not present, use the supplied corpus-building script:



```powershell

python scripts/build\_corpus.py

```



Then run the ingestion command.



\### 7. Start the API



```powershell

uvicorn app.main:app --reload

```



API address:



`http://127.0.0.1:8000`



Interactive API documentation:



`http://127.0.0.1:8000/docs`



\## API Usage



\### Health Check



\*\*GET\*\* `/health`



Example response:



```json

{

&#x20; "status": "ok"

}

```



\### Ask a Question



\*\*POST\*\* `/ask`



Example request:



```json

{

&#x20; "question": "What

