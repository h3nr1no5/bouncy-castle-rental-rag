#!/bin/sh
set -e

DB_DIR="${DB_DIR:-db}"

if [ ! -f "$DB_DIR/bm25_index.pkl" ] || [ ! -f "$DB_DIR/faiss_index.bin" ] || [ ! -f "$DB_DIR/ingest_docs.json" ]; then
    echo "Index files missing; building indexes..."
    python -c "from src.ingest import build_indexes; build_indexes()"
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
