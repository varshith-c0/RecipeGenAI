source ./.venv/bin/activate

# gthread, not sync: the workload is I/O-bound (Gemini + Qdrant round trips),
# so threads let one worker serve several requests concurrently. 3 workers x
# 4 threads = up to 12 requests in flight — tune down if Gemini 429 rates
# climb under load (check the API tier's QPM/TPM limit before raising this).
# --timeout must be >= the worst-case per-request latency (per-call timeouts
# + retries in core/gemini_client.py); 90s covers that with headroom without
# reverting to the old --timeout=30000, which disabled the watchdog entirely.
while true; do
    uv run gunicorn \
        --workers 3 --worker-class gthread --threads 4 \
        --timeout 90 --graceful-timeout 30 --keep-alive 5 \
        -b 0.0.0.0:8100 flask-app:app
done
