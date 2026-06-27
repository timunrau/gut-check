# Gut Check

Gut Check is a private, single-user IBS tracking PWA. It saves messy natural-language entries, classifies them, parses them with a local Ollama model, stores everything in SQLite, and shows simple timeline/log views.

It is not a medical diagnosis app and should not be treated as one.

## Local Hosting

Create an optional `.env` file from the example:

```bash
cp .env.example .env
```

Set at least:

```bash
APP_PASSWORD=your-password
SESSION_SECRET=a-long-random-string
APP_TIMEZONE=America/Winnipeg
```

The default parser model is `gemma4:e4b`. It is a better fit than the old
`qwen2.5:1.5b` default for cleaning messy voice-note style entries into
structured JSON, while still being realistic for a 16GB RAM host that runs
other Docker apps. If the host starts swapping or other containers suffer,
use `qwen3:4b` instead:

```bash
OLLAMA_MODEL=qwen3:4b
```

Start Ollama and pull the default model once:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:e4b
```

Run the app:

```bash
docker compose up --build
```

Open:

```text
http://SERVER_IP:8080
```

Do not expose the Ollama service directly.

## Docker Development Setup

Use Docker for normal development:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Open the Vite dev app:

```text
http://localhost:8080
```

The dev frontend proxies `/api` to the API container. FastAPI runs with reload, and Vite runs with hot reload.

Pull the Ollama model once if you have not already:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec ollama ollama pull gemma4:e4b
```

Stop dev containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

Local Python/Node setup is optional. Use it only if you need to debug outside Docker.

## Useful Commands

Try the lighter fallback model on a busy host:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:4b
OLLAMA_MODEL=qwen3:4b docker compose up --build
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

## Model Notes

Parsing is designed to run locally through Ollama. The app saves raw text first,
then asks the model to classify and clean the entry into structured JSON.

Default resource guardrails:

```text
OLLAMA_MODEL=gemma4:e4b
OLLAMA_NUM_CTX=4096
OLLAMA_KEEP_ALIVE=5m
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
```

These settings keep parsing conservative on a shared Docker host. If parse
quality is poor and the host has headroom, test a larger model manually before
making it the default. If memory pressure is visible, downgrade to `qwen3:4b`.
