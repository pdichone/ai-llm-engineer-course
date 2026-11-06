# AI Engineer Bootcamp: Course Code

All the code for the AI Engineer Bootcamp, in one repo. Each folder matches a section of the course:
`s15-vector-databases/` is the code for **Section 15**.

## Quick start

1. Install **Python 3.12**.
2. Clone this repo and copy the example keys file:
   ```bash
   git clone https://github.com/pdichone/ai-llm-engineer-course.git
   cd ai-llm-engineer-course
   cp .env.example .env        # then paste in the keys your section needs
   ```
3. Pick the setup for the section you're on (see the **Setup** column below).

### Setup A: classic sections (`pip`)

```bash
python3.12 -m venv .venv-classic
source .venv-classic/bin/activate          # Windows: .venv-classic\Scripts\activate
pip install -r requirements.txt
cd s15-vector-databases                     # the section you're on
python vector_db_llm.py
```

### Setup B: 2026 sections (`uv`)

```bash
pip install uv                              # once (or see docs.astral.sh/uv)
uv sync                                     # creates .venv with the 2026 packages
uv run s09-llm-apis/01_hello_three.py       # run from the repo root
```

The two setups use different package versions on purpose. The classic sections are pinned to the
versions used when they were recorded, and the 2026 sections use the current SDKs. Keep them in their
own virtual environments (`.venv-classic` and `.venv`).

## Course index

| Section | Folder | Setup | What's inside |
|---|---|---|---|
| 1–7 | | | Introduction, setup, Python deep dive, ML/DL and GenAI concepts (no code in this repo) |
| 8 | [`s08-llm-fundamentals/`](s08-llm-fundamentals/) | A | Transformers library, your first local LLM (L107–108) |
| 9 | [`s09-llm-apis/`](s09-llm-apis/) | **B** | LLM APIs in 2026: three providers, tokens, reasoning and effort, models, cost and caching, context windows, local vs API |
| 10 | [`s10-prompt-engineering/`](s10-prompt-engineering/) | A | First OpenAI call and prompting techniques |
| 11 | [`s11-ollama/`](s11-ollama/) | A | Ollama: CLI, REST API, Python SDK, PDF RAG, function calling, voice RAG |
| 12 | [`s12-context-memory/`](s12-context-memory/) | A | Context and memory for chatbots |
| 13 | [`s13-logging/`](s13-logging/) | A | Logging in LLM applications |
| 14 | | | RAG concepts (no code) |
| 15 | [`s15-vector-databases/`](s15-vector-databases/) | A | ChromaDB, embeddings, similarity, vector DB + LLM workflow |
| 16 | [`s16-rag-pipelines/`](s16-rag-pipelines/) | A | First RAG pipeline + Streamlit UI |
| 17 | [`s17-pdf-rag/`](s17-pdf-rag/) | A | PDF RAG system with chunking |
| 18 | [`s18-langchain/`](s18-langchain/) | A | LangChain fundamentals: prompts, loaders, splitters, chains, QA bot |
| 19 | [`s19-langchain-apps/`](s19-langchain-apps/) | A | News summarizer, YouTube summarizer, voice assistant RAG |
| 20 | [`s20-advanced-rag/`](s20-advanced-rag/) | A | Query expansion and Cohere reranking |
| 21 | [`s21-multimodal-rag/`](s21-multimodal-rag/) | A | Multimodal search and recommender RAG |
| 22 | [`s22-ai-agents/`](s22-ai-agents/) | A | AI agents, LangGraph, tools, memory, human-in-the-loop, financial report agent |
| 23 | [`s23-fine-tuning-openai/`](s23-fine-tuning-openai/) | A | Fine-tuning concepts and the OpenAI fine-tuning walkthrough |
| 24 | [`s24-fine-tuning-lora/`](s24-fine-tuning-lora/) | A | LoRA / PEFT training, API service, chatting with fine-tuned models |

New sections added later get a letter after the number they follow (for example `s15b-…`), so
folder names never change.

## Watching an older video?

Some videos were recorded before the code moved into this structure. If the video shows an old folder
or repo, here's where that code lives now:

| The video shows | The code is now in |
|---|---|
| `intro/` | `s08-llm-fundamentals/` |
| `openai-intro/` | `s10-prompt-engineering/` |
| `ollama-fundamentals` repo | `s11-ollama/` |
| `context-memory/` | `s12-context-memory/` |
| `logging-llm/` | `s13-logging/` |
| `vector-databases-embeddings/` | `s15-vector-databases/` |
| `rag_fundamentals/` (`simple_rag.py`, `rag_streamlit.py`) | `s16-rag-pipelines/` |
| `rag_fundamentals/` (`rag_pdf_simple.py`) | `s17-pdf-rag/` |
| `langchain-fundamentals/` (prompts, loaders, splitters, chains, `chatbot_qa.py`) | `s18-langchain/` |
| `langchain-fundamentals/` (news, YouTube, voice assistant) | `s19-langchain-apps/` |
| `advanced-rag/` | `s20-advanced-rag/` |
| `multimodal-rag` repo | `s21-multimodal-rag/` |
| `ai-agents-course` repo | `s22-ai-agents/` |
| `fine-tuning-openai/` | `s23-fine-tuning-openai/` |
| `fine-tuning-lora/` | `s24-fine-tuning-lora/` |

The file names inside each folder haven't changed.

## API keys

Put your keys in a `.env` file at the repo root (start from `.env.example`). Each section only needs
the keys it uses. Never commit `.env`.

## Updates

- **November 2026:** all course code now lives in this one repo, organised by section. New Section 9
  code for the 2026 lessons uses `uv` (Setup B).
- **November 2026:** OpenAI shut down `gpt-4` and `gpt-3.5-turbo`. The code that used them now uses
  `gpt-4o-mini`. Classic dependencies are pinned to the versions used in the lectures, because newer
  LangChain releases moved several imports used in the course. The AI Agents human-in-the-loop example
  was fixed for newer `langgraph-checkpoint-sqlite` versions.
