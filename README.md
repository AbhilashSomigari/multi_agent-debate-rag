# Multi-Agent Debate RAG

A multi-agent, debate-style Retrieval-Augmented Generation (RAG) pipeline built with
[LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com/),
running entirely locally (no API keys required).

## How it works

1. **Ingest & index** – documents in `docs/` (`.txt`, `.md`, `.pdf`) are chunked and embedded
   into a persistent [Chroma](https://www.trychroma.com/) vector store.
2. **Dual retrieval** – two independent agents retrieve context with different strategies:
   - **Agent A** – narrow, top-6 chunks, concise/technical style.
   - **Agent B** – broad, top-14 chunks, query rewritten to surface counterpoints and edge cases.
3. **Answer** – each agent drafts an answer grounded only in its own retrieved context, citing
   chunk ids like `[c12]`.
4. **Cross-examination** – each agent critiques the other's answer, checking whether claims are
   actually supported by the cited chunks.
5. **Revise** – each agent revises its answer based on the critique it received.
6. **Judge** – a judge agent synthesizes a final answer from both revised answers, both contexts,
   and both critiques, resolving contradictions explicitly.
7. **Verify** – a lightweight citation checker flags any factual sentence missing a `[c#]`
   citation and triggers a rewrite pass if needed.

## Setup

### 1. Install Ollama and pull models

```bash
ollama pull qwen2.5
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add documents

Drop your `.txt`, `.md`, or `.pdf` files into `docs/` (a placeholder `docs/sample.md` is
included so the script runs out of the box).

## Usage

```bash
python multi_agent_debate_rag.py --question "Your question here"
```

### Options

| Flag             | Default            | Description                              |
|------------------|---------------------|------------------------------------------|
| `--docs_dir`     | `./docs`            | Folder with source documents             |
| `--persist_dir`  | `./chroma_db`       | Chroma persistent index directory        |
| `--question`     | *(required)*        | The question to answer                   |
| `--model`        | `qwen2.5`           | Ollama chat model                        |
| `--embed_model`  | `nomic-embed-text`  | Ollama embedding model                   |
| `--rebuild`      | `false`             | Force rebuild of the vector index        |

## Notes

- The Chroma index (`chroma_db/`) is built on first run and persisted; use `--rebuild` to
  regenerate it after changing documents.
- Citation enforcement is intentionally simple/pragmatic — a sentence-level heuristic, not a
  guarantee of factual correctness.

## License

MIT — see [LICENSE](LICENSE).
