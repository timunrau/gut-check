# Gut Check

Gut Check is a private, single-user IBS tracking PWA. It saves messy natural-language entries, classifies them, parses them with a local Ollama model, stores everything in SQLite, and shows simple timeline/log views.

It is not a medical diagnosis app and should not be treated as one.

## Local Hosting

Create `.env`, start the stack detached, and pull the default model:

```bash
test -f .env || cp .env.example .env
docker compose up -d --build ollama
docker compose exec ollama ollama pull qwen3:4b
docker compose up -d --build
```

Open:

```text
http://SERVER_IP:18080
```

Do not expose the Ollama service directly.

For Android standalone/PWA mode, install the app from a secure origin. Chrome
usually treats `http://SERVER_IP:18080` as a normal website shortcut, not a true
standalone PWA. Put the web service behind HTTPS, then use Chrome's install/add
to home screen flow from that HTTPS URL.

Set at least `APP_PASSWORD`, `SESSION_SECRET`, and `APP_TIMEZONE` in `.env`
before using the app. The default web port is `18080`; override it with
`WEB_PORT=19090` in `.env` if needed.

Garmin sync runs once at API startup when saved Garmin tokens exist, then runs
nightly in the app timezone. Configure it with:

```text
GARMIN_AUTO_SYNC_ENABLED=true
GARMIN_SYNC_TIME=03:15
GARMIN_SYNC_DAYS=14
```

`GARMIN_SYNC_TIME` is a 24-hour `HH:MM` local time based on `APP_TIMEZONE`.
The nightly sync uses the same saved Garmin tokens as the manual sync button.
Set `GARMIN_AUTO_SYNC_ENABLED=false` to use manual Garmin sync only.

The default parser model is `qwen3:4b`. It is a safer default for Docker
Desktop and shared hosts than larger models that can be killed while loading.
If the host has enough memory headroom and parse quality needs improvement,
try `gemma4:e4b` by setting `OLLAMA_MODEL=gemma4:e4b` in `.env`, then pull it:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:e4b
docker compose up -d --build
```

## Docker Development Setup

Use Docker for normal development:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Open the Vite dev app:

```text
http://localhost:18080
```

The dev frontend proxies `/api` to the API container. FastAPI runs with reload, and Vite runs with hot reload.

Pull the Ollama model once if you have not already:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec ollama ollama pull qwen3:4b
```

Stop dev containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

Local Python/Node setup is optional. Use it only if you need to debug outside Docker.

## Useful Commands

Try the heavier model only if the host has memory headroom:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:e4b
OLLAMA_MODEL=gemma4:e4b docker compose up --build
```

Build frontend:

```bash
cd frontend
npm run build
```

Check Compose config:

```bash
docker compose config
```

Stop the stack:

```bash
docker compose down
```

## Data

The SQLite database is stored at:

```text
./data/gutcheck.db
```

Raw entries are saved before parsing, so Ollama failures should not prevent capture.
The dump page returns after the raw entry is committed, then the API continues
AI parsing in the background. It is safe to close the browser/PWA after the
saved pending entry appears. Stopping the backend or Docker stack will interrupt
in-progress parsing, but the raw entry remains saved and can be reparsed from
Logs.

## Model Notes

Parsing is designed to run locally through Ollama. The app saves raw text first,
then asks the model in the API process to classify and clean the entry into
structured JSON.

Default resource guardrails:

```text
OLLAMA_MODEL=qwen3:4b
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=1024
OLLAMA_TIMEOUT_SECONDS=60
OLLAMA_KEEP_ALIVE=5m
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
```

These settings keep parsing conservative on a shared Docker host. If parse
quality is poor and the host has headroom, test a larger model manually before
making it the default.

## License

MIT-0. See [LICENSE](LICENSE).
