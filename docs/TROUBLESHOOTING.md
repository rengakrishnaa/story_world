# StoryWorld Troubleshooting

Common issues and solutions.

---

## Video Issues

### Black or blank video

**Cause:** Worker is using `stub` backend, which produces 64×64 black placeholder frames.

**Solution:**
1. Set `DEFAULT_BACKEND=veo` or `DEFAULT_BACKEND=svd` in `.env`
2. Remove duplicate `DEFAULT_BACKEND` entries (last value wins)
3. Restart the main server (uvicorn)
4. Run a **new** simulation (old episodes have beats with backend already set)
5. If worker still shows `backend=stub`, clear old jobs: `python clear_job_queue.py`

---

### Worker receives backend=stub despite DEFAULT_BACKEND=veo

**Cause:** Old jobs in Redis queue from previous runs; Redis is FIFO.

**Solution:**
1. Run `python clear_job_queue.py` to clear the job queue
2. Restart main server
3. Run a new simulation

---

### Video shows zoom/camera motion instead of object action

**Cause:** SDXL + SparseMotionEngine interpolates between two similar keyframes; produces camera-like motion.

**Solution:** Use Veo when credits allow; or enhance keyframe generation to produce distinct phases (arm up vs arm down, etc.). This is a known limitation of SDXL motion interpolation.

---

## Outcome Issues

### EPISTEMICALLY_BLOCKED when expecting goal_achieved

**Cause:** Observer returned uncertain with missing evidence; intent was classified as perceptual (requires_visual_verification=true).

**Solution:** 
- Ensure goal specifies physical parameters (mass, dimensions, friction, gravity)
- Closed-form intents now accept solver-only success when observer has insufficient evidence
- If still blocked, check intent classifier returns `requires_visual_verification=false` for your goal

---

### outcome=epistemically_incomplete

**Cause:** Intent requires perceptual evidence; observer could not extract it.

**Solution:** Add more physical parameters to goal; or use API override `requires_visual_verification=false` if appropriate for your use case.

---

### outcome=uncertain_termination

**Cause:** Observer returned uncertain despite evidence.

**Solution:** Check observer logs (Gemini 429? Ollama timeout?); ensure observer is running and configured.

---

## Infrastructure Issues

### Worker not picking up jobs

**Cause:** Redis URL wrong, queue names mismatch, network.

**Solution:**
1. Verify `REDIS_URL` in both main server and worker `.env`
2. Verify `JOB_QUEUE` and `RESULT_QUEUE` match exactly
3. Test Redis: `python debug_queue_status.py`
4. Ensure worker has network access to Upstash Redis

---

### ResultConsumer not receiving results

**Cause:** Worker pushes to different result queue; Redis URL mismatch.

**Solution:**
1. Main server and worker must use same `RESULT_QUEUE` (e.g., `storyworld:gpu:results`)
2. Check ResultConsumer logs: `[ResultConsumer] Polling queue '...'`
3. Verify worker pushes to same queue after render

---

### Artifacts not uploading to R2

**Cause:** R2 credentials, endpoint, or permissions.

**Solution:**
1. Verify `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
2. Check worker logs for upload errors
3. Ensure R2 bucket allows writes from worker

---

### Observer times out or returns 429

**Cause:** Gemini rate limit; Ollama not running or slow.

**Solution:**
1. Set `OBSERVER_FALLBACK_ENABLED=true` to use Ollama when Gemini 429
2. Run `ollama run llava` to preload vision model
3. Increase Ollama timeout if needed
4. For closed-form intents, solver-only success will still apply when observer fails

---

## Configuration Issues

### Duplicate env vars

**Cause:** `.env` has multiple entries for same variable; last one wins.

**Solution:** Keep only one entry per variable; remove duplicates.

---

### Main server uses old config

**Cause:** uvicorn loads `.env` at startup; changes require restart.

**Solution:** Restart uvicorn after changing `.env`.

---

## Debugging

### Check DEFAULT_BACKEND at startup

Main server logs: `[main] Loaded .env ..., DEFAULT_BACKEND=veo`

### Check beat backend when planning

Narrative planner logs: `[narrative_planner] DEFAULT_BACKEND=veo -> beat backend=veo`

### Check queue status

```bash
python debug_queue_status.py
```

### Check worker backend

Worker logs: `[gpu-worker] job=... backend=veo ...`

### Inspect episode result

`GET /episodes/{id}/result?include_video=false`

---

## Getting Help

- **Documentation:** [docs/README.md](README.md)
- **API Reference:** [docs/API_REFERENCE.md](API_REFERENCE.md)
- **Configuration:** [docs/CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)
