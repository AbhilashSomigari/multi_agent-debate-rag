# Sample Knowledge Base Document

This is a placeholder document so the RAG pipeline has something to index out of the box.
Replace or add your own `.txt`, `.md`, or `.pdf` files in this `docs/` folder.

## About Retrieval-Augmented Generation (RAG)

RAG combines a retrieval step (searching a vector database for relevant text chunks)
with a generation step (an LLM writing an answer grounded in those chunks). This reduces
hallucination because the model is instructed to cite the specific chunks it used.

## About Multi-Agent Debate

In a multi-agent debate setup, two or more agents independently answer the same question,
then critique each other's reasoning before a judge agent synthesizes a final, better-grounded
answer. This technique tends to surface unsupported claims and gaps in evidence that a single
agent might miss.
