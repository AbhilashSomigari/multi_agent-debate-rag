"""
Multi-Agent Debate RAG (Ollama + LangGraph)

A multi-agent debate system using Retrieval-Augmented Generation (RAG) with LangGraph and Ollama.:
- Two independent RAG agents (A and B) that retrieve with different strategies
- Debate round (cross-exam) where each attacks the other's grounding
- Judge that synthesizes a final answer
- Verifier that forces citations (simple, pragmatic enforcement)

Requirements (pip):
  pip install -U langgraph langchain langchain-community chromadb pypdf

Ollama:
  - Install Ollama and pull models:
      ollama pull qwen2.5
      ollama pull nomic-embed-text

Run:
  python multi_agent_debate_rag.py --docs_dir ./docs --question "Your question here"

Notes:
- knowledge base files in ./docs (txt, md, pdf).
- First run will build a Chroma index in ./chroma_db (persistent).
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END

# Optional PDF support
try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False


# -----------------------------
# Utilities: load & chunk docs
# -----------------------------
def load_docs_from_dir(docs_dir: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    for root, _, files in os.walk(docs_dir):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[1]

            if ext in [".txt", ".md"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": path}))
            elif ext == ".pdf":
                if not HAVE_PYPDF:
                    print(f"[warn] Skipping PDF (install pypdf): {path}", file=sys.stderr)
                    continue
                try:
                    reader = PdfReader(path)
                    text_parts = []
                    for page in reader.pages:
                        t = page.extract_text() or ""
                        if t.strip():
                            text_parts.append(t)
                    text = "\n\n".join(text_parts).strip()
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": path}))
                except Exception as e:
                    print(f"[warn] Failed to read PDF {path}: {e}", file=sys.stderr)

    if not docs:
        raise RuntimeError(f"No supported documents found in {docs_dir}. Add .txt/.md/.pdf files.")
    return docs


def chunk_docs(docs: List[Document], chunk_size: int = 900, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: List[Document] = []
    cid = 0
    for d in docs:
        for c in splitter.split_documents([d]):
            meta = dict(c.metadata or {})
            meta["chunk_id"] = f"c{cid}"
            cid += 1
            chunks.append(Document(page_content=c.page_content, metadata=meta))
    return chunks


def build_or_load_vectorstore(
    docs_dir: str,
    persist_dir: str,
    embed_model: str,
    rebuild: bool,
) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OllamaEmbeddings(model=embed_model)

    # If rebuild requested, wipe Chroma directory
    if rebuild:
        for name in os.listdir(persist_dir):
            p = os.path.join(persist_dir, name)
            try:
                if os.path.isdir(p):
                    import shutil
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass

    # Try load existing
    try:
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        # Heuristic: if empty, rebuild
        if vs._collection.count() == 0:
            raise RuntimeError("Empty index")
        return vs
    except Exception:
        # Build
        raw_docs = load_docs_from_dir(docs_dir)
        chunks = chunk_docs(raw_docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        vs.persist()
        return vs


# -----------------------------
# RAG + agents
# -----------------------------
def format_context(docs: List[Document], max_chars: int = 9000) -> str:
    """
    Build context with chunk ids + sources so citations can reference chunk ids.
    """
    parts = []
    total = 0
    for d in docs:
        cid = d.metadata.get("chunk_id", "unknown")
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.strip()
        block = f"[{cid}] (source: {src})\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


def retrieve(vs: Chroma, query: str, k: int) -> List[Document]:
    # Chroma similarity_search returns Documents with metadata preserved
    return vs.similarity_search(query, k=k)


@dataclass
class AgentConfig:
    name: str
    k: int
    style: str  # used to force diversity


def make_answer(llm: ChatOllama, question: str, context_docs: List[Document], agent_cfg: AgentConfig) -> str:
    context = format_context(context_docs)
    system = (
        f"You are Answerer-{agent_cfg.name}. Use ONLY the provided context. "
        "Do not invent facts. If context is insufficient, say what is missing.\n\n"
        "CITATION RULE: Every important claim must include at least one citation like [c12]. "
        "Citations must refer to chunk ids from the context.\n\n"
        f"Answering style: {agent_cfg.style}"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write a clear answer with citations."
    )
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content


def make_critique(llm: ChatOllama, question: str, my_context_docs: List[Document], other_answer: str, who: str) -> str:
    my_context = format_context(my_context_docs)
    system = (
        f"You are Debater-{who}. Your job is to critique the OTHER answer for grounding and completeness.\n"
        "Rules:\n"
        "- Identify any claim that is unsupported or weakly supported.\n"
        "- If you disagree, point to EXACT chunk ids from YOUR context.\n"
        "- If the other answer cites a chunk, check whether it truly supports the claim.\n"
        "- Propose a corrected, better-grounded alternative when possible.\n"
        "- Keep it concise and evidence-driven."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"YOUR CONTEXT:\n{my_context}\n\n"
        f"OTHER ANSWER:\n{other_answer}\n\n"
        "Produce: (1) Key issues list, (2) Corrections with citations, (3) Missing info."
    )
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content


def make_revised_answer(
    llm: ChatOllama,
    question: str,
    context_docs: List[Document],
    original_answer: str,
    critique_from_other: str,
    agent_cfg: AgentConfig,
) -> str:
    context = format_context(context_docs)
    system = (
        f"You are Answerer-{agent_cfg.name} revising your answer after critique.\n"
        "Rules:\n"
        "- Use ONLY provided context.\n"
        "- Fix unsupported claims.\n"
        "- Keep citations for key claims using chunk ids like [c7].\n"
        "- If something is missing from context, state that explicitly.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"YOUR ORIGINAL ANSWER:\n{original_answer}\n\n"
        f"CRITIQUE YOU RECEIVED:\n{critique_from_other}\n\n"
        "Now produce a revised answer with strong grounding and citations."
    )
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content


def judge_synthesize(
    llm: ChatOllama,
    question: str,
    ctx_a: List[Document],
    ctx_b: List[Document],
    ans_a: str,
    ans_b: str,
    crit_a: str,
    crit_b: str,
) -> str:
    # Provide both contexts so judge can cite from either set of chunk ids.
    context = "CONTEXT-A:\n" + format_context(ctx_a) + "\n\nCONTEXT-B:\n" + format_context(ctx_b)
    system = (
        "You are the Judge. Your job is to produce the best final answer grounded in evidence.\n"
        "Rules:\n"
        "- Use ONLY the provided contexts.\n"
        "- Prefer claims supported by multiple chunks or strongest chunks.\n"
        "- Resolve contradictions explicitly.\n"
        "- Every key claim must include citations like [c12].\n"
        "- If the contexts are insufficient, say what would be needed.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{context}\n\n"
        f"ANSWER-A:\n{ans_a}\n\n"
        f"ANSWER-B:\n{ans_b}\n\n"
        f"CRITIQUE-A (A critiques B):\n{crit_a}\n\n"
        f"CRITIQUE-B (B critiques A):\n{crit_b}\n\n"
        "Write the FINAL answer with citations."
    )
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content


# -----------------------------
# Verifier (simple enforcement)
# -----------------------------
CITE_RE = re.compile(r"\[c\d+\]")

def count_sentences_without_cites(text: str) -> int:
    # Naive: split into sentences; count those that look factual and lack citations
    # This is deliberately simple; good enough for a demo.
    sentences = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    bad = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # ignore headings / bullet titles
        if len(s) < 12:
            continue
        # If sentence contains any citation, OK.
        if CITE_RE.search(s):
            continue
        # If it's explicitly uncertainty / missing info, allow.
        if re.search(r"\b(insufficient|not enough|unknown|cannot determine|missing)\b", s, re.I):
            continue
        bad += 1
    return bad


def revise_for_citations(llm: ChatOllama, question: str, ctx_a: List[Document], ctx_b: List[Document], draft: str) -> str:
    context = "CONTEXT-A:\n" + format_context(ctx_a) + "\n\nCONTEXT-B:\n" + format_context(ctx_b)
    system = (
        "You are a strict verifier/editor.\n"
        "Task: revise the draft so that every meaningful factual sentence has at least one citation [c#].\n"
        "Rules:\n"
        "- Use ONLY the provided contexts.\n"
        "- If a sentence cannot be supported, rewrite it to be uncertainty-aware or remove it.\n"
        "- Keep the answer clean and readable.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{context}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        "Return the revised answer with correct citations."
    )
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content


# -----------------------------
# LangGraph State + nodes
# -----------------------------
class DebateState(TypedDict, total=False):
    question: str
    # retrieval results
    docs_a: List[Document]
    docs_b: List[Document]
    # answers
    ans_a: str
    ans_b: str
    # critiques
    crit_a: str  # A critiques B
    crit_b: str  # B critiques A
    # revised answers (optional; we do 1 round)
    ans_a_rev: str
    ans_b_rev: str
    # final
    final: str


def make_graph(vs: Chroma, llm: ChatOllama):
    cfg_a = AgentConfig(name="A", k=6, style="Be concise, technical, and structured.")
    cfg_b = AgentConfig(name="B", k=14, style="Be thorough; surface edge cases and alternative interpretations.")

    def retrieve_a(state: DebateState) -> DebateState:
        q = state["question"]
        docs = retrieve(vs, q, k=cfg_a.k)
        return {"docs_a": docs}

    def retrieve_b(state: DebateState) -> DebateState:
        q = state["question"]
        # Force diversity: add a different “angle” query suffix
        q2 = q + " (include counterpoints, exceptions, and definitions)"
        docs = retrieve(vs, q2, k=cfg_b.k)
        return {"docs_b": docs}

    def answer_a(state: DebateState) -> DebateState:
        return {"ans_a": make_answer(llm, state["question"], state["docs_a"], cfg_a)}

    def answer_b(state: DebateState) -> DebateState:
        return {"ans_b": make_answer(llm, state["question"], state["docs_b"], cfg_b)}

    def critique_a(state: DebateState) -> DebateState:
        # A critiques B using A's context
        return {"crit_a": make_critique(llm, state["question"], state["docs_a"], state["ans_b"], who="A")}

    def critique_b(state: DebateState) -> DebateState:
        # B critiques A using B's context
        return {"crit_b": make_critique(llm, state["question"], state["docs_b"], state["ans_a"], who="B")}

    def revise_a(state: DebateState) -> DebateState:
        return {"ans_a_rev": make_revised_answer(llm, state["question"], state["docs_a"], state["ans_a"], state["crit_b"], cfg_a)}

    def revise_b(state: DebateState) -> DebateState:
        return {"ans_b_rev": make_revised_answer(llm, state["question"], state["docs_b"], state["ans_b"], state["crit_a"], cfg_b)}

    def judge(state: DebateState) -> DebateState:
        ans_a = state.get("ans_a_rev", state["ans_a"])
        ans_b = state.get("ans_b_rev", state["ans_b"])
        final = judge_synthesize(
            llm,
            state["question"],
            state["docs_a"],
            state["docs_b"],
            ans_a,
            ans_b,
            state["crit_a"],
            state["crit_b"],
        )
        return {"final": final}

    def verify(state: DebateState) -> DebateState:
        draft = state["final"]
        bad = count_sentences_without_cites(draft)
        if bad == 0:
            return {}
        revised = revise_for_citations(llm, state["question"], state["docs_a"], state["docs_b"], draft)
        return {"final": revised}

    g = StateGraph(DebateState)
    g.add_node("retrieve_a", retrieve_a)
    g.add_node("retrieve_b", retrieve_b)
    g.add_node("answer_a", answer_a)
    g.add_node("answer_b", answer_b)
    g.add_node("critique_a", critique_a)
    g.add_node("critique_b", critique_b)
    g.add_node("revise_a", revise_a)
    g.add_node("revise_b", revise_b)
    g.add_node("judge", judge)
    g.add_node("verify", verify)

    # Flow
    g.set_entry_point("retrieve_a")
    g.add_edge("retrieve_a", "retrieve_b")
    g.add_edge("retrieve_b", "answer_a")
    g.add_edge("answer_a", "answer_b")
    g.add_edge("answer_b", "critique_a")
    g.add_edge("critique_a", "critique_b")
    g.add_edge("critique_b", "revise_a")
    g.add_edge("revise_a", "revise_b")
    g.add_edge("revise_b", "judge")
    g.add_edge("judge", "verify")
    g.add_edge("verify", END)

    return g.compile()

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", type=str, default="./docs", help="Folder with .txt/.md/.pdf files")
    ap.add_argument("--persist_dir", type=str, default="./chroma_db", help="Chroma persistent directory")
    ap.add_argument("--question", type=str, required=True, help="User question")
    ap.add_argument("--model", type=str, default="qwen2.5", help="Ollama chat model")
    ap.add_argument("--embed_model", type=str, default="nomic-embed-text", help="Ollama embedding model")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild vector index")
    args = ap.parse_args()

    vs = build_or_load_vectorstore(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        embed_model=args.embed_model,
        rebuild=args.rebuild,
    )

    llm = ChatOllama(model=args.model, temperature=0.2)

    app = make_graph(vs, llm)
    print(app.get_graph().draw_mermaid())

    start = time.time()
    out = app.invoke({"question": args.question})
    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(out["final"].strip())
    print("\n" + "=" * 80)
    print(f"Done in {elapsed:.1f}s | docs_dir={args.docs_dir} | index={args.persist_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
