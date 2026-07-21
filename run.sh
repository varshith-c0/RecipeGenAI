source ./.venv/bin/activate

while true; do
    uv run gunicorn --timeout=30000 -b 0.0.0.0:8100 flask-app:app
done
