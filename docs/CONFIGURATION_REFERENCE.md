# StoryWorld Configuration Reference

We use `env.example` as a template. I copy it to `.env`, fill in Redis, R2, and Gemini credentials, and never commit `.env`.

---

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `local` | `production` or `local` |
| `REALITY_COMPILER_MODE` | `true` | Neutral style, no cinematic specs, observer-driven retry |
| `USE_OBSERVER_IN_PRODUCTION` | `true` | Run observer on each successful render; populate constraints_discovered |
| `LOCAL_MODE` | `false` | Local vs distributed mode |

---

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./local.db` | SQLite or PostgreSQL URL |

---

## Redis (Job Queue)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | — | Upstash Redis URL (`rediss://...`) |
| `JOB_QUEUE` | `storyworld:gpu:jobs` | GPU job queue name |
| `RESULT_QUEUE` | `storyworld:gpu:results` | GPU result queue name |
| `GPU_JOB_QUEUE` | — | Override job queue |
| `GPU_RESULT_QUEUE` | — | Override result queue |
| `GPU_RESULT_QUEUE_PREFIX` | — | Per-episode suffix for result queue |

---

## R2/S3 Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_ENDPOINT` | — | R2/S3 endpoint |
| `S3_BUCKET` | — | Bucket name |
| `S3_ACCESS_KEY` | — | Access key |
| `S3_SECRET_KEY` | — | Secret key |
| `S3_REGION` | `auto` | Region (use `auto` for R2) |
| `ARTIFACTS_DIR` | — | Shared dir when worker and observer share a volume |

---

## Video Backends

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_BACKEND` | `veo` | Default backend: `veo`, `svd`, `animatediff`, `stub` |
| `VEO_FALLBACK_BACKEND` | `svd` | Fallback when Veo credit exhausted |
| `VEO_USE_FAST` | `false` | Use Veo fast model |
| `USE_DIFFUSION` | `false` | Enable local SDXL for SVD/AnimateDiff fallback when Veo credit exhausted |

---

## Planner

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_PLANNER` | `false` | Use mock planner (no Gemini) |
| `GEMINI_API_KEY` | — | Gemini API key |
| `GEMINI_USE_REAL` | `true` | Use real Gemini when mock planner enabled |

---

## Observer

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_OBSERVER_MODEL` | `gemini-2.5-flash-lite` | Vision model override |
| `OBSERVER_FALLBACK_ENABLED` | — | Fallback to Ollama when Gemini 429 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llava` | Vision model for fallback |

---

## Intent Classifier

| Variable | Default | Description |
|----------|---------|-------------|
| `INTENT_CLASSIFIER_MODEL` | `gemini-2.0-flash` | LLM model |
| `INTENT_CLASSIFIER_CACHE_SIZE` | `2000` | LRU cache size |
| `INTENT_CLASSIFIER_OLLAMA_MODEL` | `llama2` | Fallback text model |
| `INTENT_CLASSIFIER_OLLAMA_TIMEOUT` | `15` | Timeout in seconds |

---

## Simulation Policies

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_RETRY_MAX_ATTEMPTS` | `3` | Max retries for simulation |
| `SIM_QUALITY_MIN_CONFIDENCE` | `0.8` | Min confidence threshold |
| `SIM_COST_MAX_USD` | `10.0` | Max cost per simulation |
| `EP_RETRY_MAX_ATTEMPTS` | `2` | Max retries per episode |
| `EP_QUALITY_MIN_CONFIDENCE` | `0.7` | Episode confidence threshold |
| `EP_COST_MAX_USD` | `5.0` | Max cost per episode |

---

## Cost & Confidence

| Variable | Default | Description |
|----------|---------|-------------|
| `COST_PER_BEAT_USD` | `0.05` | Cost per beat |
| `DEFAULT_SUCCESS_CONFIDENCE` | `0.85` | Default confidence |
| `EPISODE_COMPOSE_REQUIRED_CONFIDENCE` | `0.3` | Min confidence for compose |

---

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVABILITY_MAX_ATTEMPTS` | `2` | Max re-renders for uncertain/insufficient_evidence |

---

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level |
| `DECISION_LOOP_LOG_PATH` | `decision_loop.log` | Path for decision loop logs |
| `FFMPEG_PATH` | — | Path to ffmpeg binary when not in PATH |
