"""
query_services.py
-----------------
Business logic layer for the RAG / SQL / Hybrid query pipeline.

Guardrail placement rules:
  - guard_input for /query        → inside query()         (before graph.invoke)
  - guard_input for /query/stream → inside check_query_guard(), called by the
                                    route BEFORE StreamingResponse is created.
                                    NOT inside query_stream() — by the time the
                                    generator is iterated the HTTP 200 header is
                                    already sent and a GuardrailViolation can no
                                    longer become an HTTP 400.
  - guard_output                  → inside query() on the final answer string.
                                    Not applied per-token in the stream (would
                                    corrupt the character-by-character flow).
"""

from pathlib import Path

from fastapi import UploadFile

from src.api.v1.agents.agents import graph, run_agent_stream
from src.ingestion.ingestion import run_ingestion
from src.core.guardrails import guard_input, guard_output, GuardrailViolation  # noqa: F401

# ============================================================
# Shared helpers
# ============================================================

def _build_state(query: str, session_id: str = "default") -> dict:
    """Return a fresh AgentState dict for the LangGraph graph."""
    return {
        "query": query,
        "query_type": "",
        "rag_chunks": [],
        "reranked_chunks": [],
        "sql_query": "",
        "sql_result": [],
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "session_id": session_id,
        "chat_history": [],
    }


# ============================================================
# Ingestion service
# ============================================================

UPLOAD_DIR = Path("uploads")


async def ingest_pdf(file: UploadFile) -> dict:
    """
    Save an uploaded PDF to disk and run the ingestion pipeline.

    Returns:
      {"status": "success", "doc_id": "...", "chunks_ingested": N}
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    contents = await file.read()
    file_path.write_bytes(contents)

    return run_ingestion(str(file_path))


# ============================================================
# Query service  (non-streaming)
# ============================================================

# CHANGE: Added session_id parameter to query().
# Routes that call this now pass the session_id from the request,
# which flows into _build_state and then into the graph.
def query(question: str, session_id: str = "default") -> dict:
    """
    Run the full agent graph for a regular (non-streaming) query.

    Guardrail flow:
      1. guard_input  — raises GuardrailViolation if the query is toxic.
      2. graph.invoke — RAG / SQL / Hybrid pipeline (memory nodes included).
      3. guard_output — redacts PII from the answer before returning.

    Returns:
      {"answer": str, "citations": list[str]}
    """
    # Input guard — safe to raise here, no HTTP header sent yet
    guard_input(question)

    # CHANGE: Pass session_id into _build_state
    state = _build_state(question, session_id)
    result = graph.invoke(state)

    answer = result.get("answer", "")
    if answer:
        answer = guard_output(answer)

    return {
        "answer": answer,
        "citations": result.get("citations", []),
    }


# ============================================================
# Query service  (streaming / SSE)
# ============================================================

def check_query_guard(question: str) -> None:
    """
    Run input guardrails for the streaming endpoint.

    MUST be called by the route BEFORE StreamingResponse is created.
    """
    guard_input(question)

async def query_stream(question: str, session_id: str = "default"):
    """
    Async generator that streams SSE tokens for a query.

    guard_input is intentionally NOT called here — call check_query_guard()
    in the route before creating StreamingResponse.

    Yields SSE events:
      data:{"token": "x"}          — one character at a time
      data:{"citations": [...]}    — sources block after the answer
      data:[DONE]                  — sentinel
    """
    # CHANGE: Forward session_id to run_agent_stream
    async for chunk in run_agent_stream(question, session_id):
        yield chunk