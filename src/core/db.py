import base64
import hashlib
import json
import os
import pathlib

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase

load_dotenv()


# ============================================================
# DATABASE CONNECTIONS
# ============================================================

RAG_DSN = os.getenv(
    "PG_CONNECTION_STRING",
    ""
).replace(
    "postgresql+psycopg://",
    "postgresql://"
)

TABLE_DSN = os.getenv(
    "AGENTIC_RAG_DB_URL",
    ""
).replace(
    "postgresql+psycopg://",
    "postgresql://"
)


# ============================================================
# EMBEDDINGS
# ============================================================

_EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

_embeddings = OpenAIEmbeddings(
    model=_EMBED_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    return _embeddings.embed_documents(texts)


# ============================================================
# CONNECTION POOLS
# ============================================================

_rag_pool: ConnectionPool | None = None
_table_pool: ConnectionPool | None = None


def _get_rag_pool() -> ConnectionPool:
    global _rag_pool

    if _rag_pool is None:
        _rag_pool = ConnectionPool(
            conninfo=RAG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )

    return _rag_pool


def _get_table_pool() -> ConnectionPool:
    global _table_pool

    if _table_pool is None:
        _table_pool = ConnectionPool(
            conninfo=TABLE_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )

    return _table_pool


# ============================================================
# CONNECTION HELPERS
# ============================================================

def get_rag_conn():
    """
    smart_banking_rag_db

    Used for:
    - documents
    - multimodal_chunks
    - vector search
    - embeddings
    """
    return _get_rag_pool().connection()


def get_table_conn():
    """
    smart_banking_tables

    Used for:
    - accounts
    - transactions
    - loan_accounts
    - fixed_deposits
    - credit_cards
    - card_transactions
    """
    return _get_table_pool().connection()


# Backward compatibility
def get_db_conn():
    """
    Existing ingestion code uses this.
    Point it to RAG DB.
    """
    return get_rag_conn()


# ============================================================
# DOCUMENT REGISTRY
# ============================================================

def upsert_document(
    filename: str,
    source_path: str
) -> str:

    with get_rag_conn() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO documents
                (
                    filename,
                    source_path
                )
                VALUES
                (
                    %s,
                    %s
                )
                ON CONFLICT (filename)
                DO UPDATE
                SET
                    source_path = EXCLUDED.source_path,
                    ingested_at = NOW()
                RETURNING id
                """,
                (
                    filename,
                    source_path
                )
            )

            row = cur.fetchone()

        conn.commit()

    return str(row["id"])


# ============================================================
# CHUNK STORAGE
# ============================================================

def store_chunks(
    chunks: list[dict],
    doc_id: str
) -> int:

    if not chunks:
        return 0

    all_embeddings = _embed_texts(
        [chunk["content"] for chunk in chunks]
    )

    _DEDICATED_COLUMNS = {
        "content_type",
        "element_type",
        "section",
        "page_number",
        "source_file",
        "position",
        "image_base64",
    }

    rows_inserted = 0

    with get_rag_conn() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM multimodal_chunks
                WHERE doc_id = %s::uuid
                """,
                (doc_id,)
            )

            for chunk, embedding in zip(chunks, all_embeddings):

                meta = chunk["metadata"]

                img_b64 = meta.get("image_base64")

                image_path = None
                mime_type = None

                if img_b64:

                    image_bytes = base64.b64decode(img_b64)

                    img_dir = pathlib.Path("data/images")

                    img_dir.mkdir(parents=True, exist_ok=True)

                    img_hash = hashlib.sha256(
                        image_bytes
                    ).hexdigest()[:16]

                    img_file = (
                        img_dir /
                        f"{doc_id}_{img_hash}.png"
                    )

                    img_file.write_bytes(image_bytes)

                    image_path = str(img_file)
                    mime_type = "image/png"

                embedding_str = (
                    "[" +
                    ",".join(str(v) for v in embedding) +
                    "]"
                )

                clean_meta = {
                    k: v
                    for k, v in meta.items()
                    if k not in _DEDICATED_COLUMNS
                }

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks
                    (
                        doc_id,
                        chunk_type,
                        element_type,
                        content,
                        image_path,
                        mime_type,
                        page_number,
                        section,
                        source_file,
                        position,
                        embedding,
                        metadata
                    )
                    VALUES
                    (
                        %s::uuid,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::vector,
                        %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk["content_type"],
                        meta.get("element_type"),
                        chunk["content"],
                        image_path,
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position"))
                        if meta.get("position")
                        else None,
                        embedding_str,
                        json.dumps(clean_meta),
                    ),
                )

                rows_inserted += 1

        conn.commit()

    return rows_inserted


def get_sql_database():

    db_url = os.getenv("AGENTIC_RAG_DB_URL")

    return SQLDatabase.from_uri(
        db_url,
        include_tables=[
            "accounts",
            "transactions",
            "loan_accounts",
            "fixed_deposits",
            "credit_cards",
            "card_transactions"
        ],
        sample_rows_in_table_info=2
    )
