#!/bin/bash
set -e

DB_HOST="${AQA_DB_HOST:-mysql}"
DB_PORT="${AQA_DB_PORT:-3306}"
DB_USER="${AQA_DB_USER:-root}"
DB_PASS="${AQA_DB_PASSWORD:-automateqa}"
DB_NAME="${AQA_DB_NAME:-automateqa}"

echo "==> Waiting for MySQL at ${DB_HOST}:${DB_PORT}..."
for i in $(seq 1 30); do
    if python3 -c "
import pymysql, sys
try:
    pymysql.connect(host='${DB_HOST}', port=${DB_PORT}, user='${DB_USER}', password='${DB_PASS}', database='${DB_NAME}')
    sys.exit(0)
except Exception as e:
    print(f'    Attempt ${i}: {e}', file=sys.stderr)
    sys.exit(1)
"; then
        echo "==> MySQL is ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> ERROR: MySQL not available after 30 attempts, exiting."
        exit 1
    fi
    sleep 2
done

echo "==> Running database migrations..."
alembic -c server/alembic/alembic.ini upgrade head

echo "==> Starting AutoMateQA Dashboard API..."

# Reload mode: watch source files and restart on changes.
# Forces workers=1 (uvicorn --reload is incompatible with multiple workers).
if [ "${AQA_RELOAD:-false}" = "true" ]; then
    echo "==> Hot-reload enabled (workers forced to 1)"
    exec uvicorn server.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --reload \
        --reload-dir /app/engine \
        --reload-dir /app/server \
        --log-level "${AQA_LOG_LEVEL:-info}"
else
    exec uvicorn server.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers "${AQA_WORKERS:-1}" \
        --log-level "${AQA_LOG_LEVEL:-info}"
fi
