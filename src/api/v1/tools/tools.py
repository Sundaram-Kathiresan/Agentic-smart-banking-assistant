from langchain_core.tools import tool
from src.core.db import get_rag_conn, get_table_conn, _embed_texts

# ==================================================
# EMBEDDING HELPER
# ==================================================

def get_embedding(query: str):
    return _embed_texts([query])[0]

# ==================================================
# VECTOR SEARCH
# ==================================================

def vector_search(query: str, top_k: int = 5):
    embedding = get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    with get_rag_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    chunk_type,
                    page_number,
                    source_file,
                    1 - (embedding <=> %s::vector) AS score
                FROM multimodal_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, top_k),
            )
            return cur.fetchall()

# ==================================================
# FTS SEARCH
# ==================================================

def fts_search(query: str, top_k: int = 5):
    with get_rag_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    chunk_type,
                    page_number,
                    source_file,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', %s)
                    ) AS score
                FROM multimodal_chunks
                WHERE
                    to_tsvector('english', content)
                    @@
                    plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return cur.fetchall()

# ==================================================
# HYBRID SEARCH (RRF)
# ==================================================

def hybrid_search(query: str, k: int):
    vector_docs = vector_search(query, 5)
    fts_docs = fts_search(query, 5)
    scores = {}
    documents = {}
    k = 60
    for rank, row in enumerate(vector_docs):
        doc_id = str(row["id"])
        documents[doc_id] = row
        scores.setdefault(doc_id, 0)
        scores[doc_id] += 1 / (k + rank)

    for rank, row in enumerate(fts_docs):
        doc_id = str(row["id"])
        documents[doc_id] = row
        scores.setdefault(doc_id, 0)
        scores[doc_id] += 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [documents[doc_id] for doc_id, _ in ranked]

# ==================================================
# SQL EXECUTOR
# ==================================================

FORBIDDEN = {
    "insert",
    "update",
    "delete",
    "drop",
    "truncate",
    "alter",
    "grant",
    "revoke",
    "create",
}

def execute_sql(sql: str):
    sql_lower = sql.lower()
    for keyword in FORBIDDEN:
        if keyword in sql_lower:
            raise Exception(f"Unsafe SQL detected: {keyword}")
    with get_table_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return rows