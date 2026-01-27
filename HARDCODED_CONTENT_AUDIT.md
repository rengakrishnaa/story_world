# Hardcoded Content Audit & Fix Report

## 🚨 CRITICAL ISSUE FOUND & FIXED

### Issue: Hardcoded Mock Planner in main.py

**Location:** `main.py` line 80

**Problem:**
```python
planner = ProductionNarrativePlanner(
    world_id=runtime.world_id,
    redis_client=redis_store.redis,   
    use_mock=True,  # ❌ HARDCODED - Always uses mock!
)
```

**Impact:**
- **ALL episodes** would use mock planner regardless of user input
- **ALL videos** would be about "Saitama vs Monster" instead of user's prompt
- User's intent completely ignored
- Production API would return test content

**Fix Applied:**
```python
# Use environment variable to control mock vs real planner
use_mock = os.getenv("USE_MOCK_PLANNER", "false").lower() == "true"

planner = ProductionNarrativePlanner(
    world_id=runtime.world_id,
    redis_client=redis_store.redis,   
    use_mock=use_mock,  # ✅ FIXED: Now respects env variable
)
```

---

## Environment Variable Control

### Production Setup (Real User Prompts)
```bash
# In .env file
USE_MOCK_PLANNER=false  # ✅ Uses real Gemini API
GEMINI_API_KEY=your_actual_key
```

**Behavior:**
- ✅ User's prompt sent to Gemini API
- ✅ AI generates custom episode plan
- ✅ Videos match user's intent
- ✅ Characters/locations from user's world

### Testing Setup (Mock Content)
```bash
# In .env file
USE_MOCK_PLANNER=true  # Uses hardcoded Saitama content
```

**Behavior:**
- ⚠️ Ignores user's prompt
- ⚠️ Returns hardcoded "One Punch Battle" episode
- ⚠️ 3 beats about Saitama vs Monster
- ⚠️ Useful only for testing pipeline

---

## Mock Content Details

When `USE_MOCK_PLANNER=true`, the system returns:

### Hardcoded Episode Plan
```json
{
  "title": "One Punch Battle: Episode 1",
  "total_duration_min": 8,
  "acts": [
    {
      "name": "Act 1: Monster Approaches",
      "summary": "Saitama faces Monster King Orochi on a ruined rooftop.",
      "scenes": [
        {
          "id": "scene-1",
          "title": "Rooftop Standoff",
          "beats": [
            {
              "id": "beat-1",
              "description": "Wide shot: ruined city skyline, massive monster approaches",
              "duration": 12,
              "characters": ["Saitama", "Genos"],
              "location": "rooftop"
            },
            {
              "id": "beat-2",
              "description": "Genos warns Saitama: 'Master, this monster is S-class!'",
              "duration": 8,
              "characters": ["Genos", "Saitama"],
              "location": "rooftop"
            },
            {
              "id": "beat-3",
              "description": "Saitama yawns: 'Oh? Looks kinda strong.'",
              "duration": 6,
              "characters": ["Saitama"],
              "location": "rooftop"
            }
          ]
        }
      ]
    }
  ]
}
```

**Location:** `agents/narrative_planner.py` lines 142-180

---

## Other Hardcoded Content (Safe)

### 1. Fallback Gradient Frames
**Location:** `agents/motion/sparse_motion_engine.py`

**Purpose:** When Veo/NanoBanana APIs fail, creates colored gradient frames

**Impact:** ✅ Safe - only used as fallback, not default

**Example:**
```python
def _create_gradient_frame(width, height, color1, color2):
    # Creates gradient from color1 to color2
    # Used when API unavailable
```

### 2. PIPELINE_VALIDATE Mode
**Location:** `agents/backends/animatediff_backend.py`

**Purpose:** Testing mode that skips real rendering

**Control:** `PIPELINE_VALIDATE=false` in .env (default)

**Impact:** ✅ Safe - disabled in production

### 3. Test Files
**Location:** `tests/test_beat_observer.py`, `models/world.py` (examples)

**Purpose:** Unit tests and documentation examples

**Impact:** ✅ Safe - not used in production code

---

## Verification Steps

### ✅ Step 1: Check .env Configuration
```bash
# Your current .env
USE_MOCK_PLANNER=false  # ✅ Correct for production
GEMINI_API_KEY=your_gemini_api_key_here  # ⚠️ Replace with real key
```

### ✅ Step 2: Test Real Prompt
```bash
POST /episodes
{
  "world_id": "my_world",
  "intent": "A dragon attacks a medieval castle"
}

POST /episodes/{id}/plan
```

**Expected Behavior:**
- ✅ Gemini API called with "A dragon attacks a medieval castle"
- ✅ Episode plan about dragons and castles
- ✅ NO Saitama content

### ✅ Step 3: Test Mock Mode (Optional)
```bash
# Set in .env
USE_MOCK_PLANNER=true

POST /episodes/{id}/plan
```

**Expected Behavior:**
- ⚠️ Returns Saitama content (ignores user prompt)
- ⚠️ Useful only for testing pipeline without API costs

---

## Production Safety Checks

### 1. Environment Variable Protection
```python
# In narrative_planner.py lines 204-207
env = os.getenv("ENV", "local").lower()

if env == "production" and use_mock:
    raise RuntimeError(
        "USE_MOCK_PLANNER=true is forbidden in production"
    )
```

**Protection:** If `ENV=production` and `USE_MOCK_PLANNER=true`, server crashes on startup

### 2. Logging Warnings
```python
# In narrative_planner.py line 130
if not self.use_real_api:
    logger.warning("🧪 Narrative planner running in MOCK mode")
```

**Protection:** Console shows warning if mock mode active

---

## Recommended .env Settings

### For Production (Real Videos)
```bash
# Planner
USE_MOCK_PLANNER=false
GEMINI_API_KEY=your_real_gemini_key

# Environment
ENV=production

# Other settings
USE_DIFFUSION=true
DEFAULT_BACKEND=animatediff
```

### For Development (Free Testing)
```bash
# Planner
USE_MOCK_PLANNER=true  # Uses Saitama mock content
# GEMINI_API_KEY not needed

# Environment
ENV=local

# Other settings
USE_DIFFUSION=true
DEFAULT_BACKEND=animatediff
```

---

## Summary

### ✅ Fixed Issues
1. **main.py line 80** - Removed hardcoded `use_mock=True`
2. Now respects `USE_MOCK_PLANNER` environment variable
3. Production mode blocks mock planner

### ⚠️ Remaining Hardcoded Content (Intentional)
1. Mock planner content (Saitama) - Only used when `USE_MOCK_PLANNER=true`
2. Fallback gradient frames - Only used when APIs fail
3. Test fixtures - Not used in production

### ✅ Your Configuration
```bash
USE_MOCK_PLANNER=false  # ✅ Will use real Gemini API
```

**Result:** Your videos will match your prompts, not hardcoded Saitama content!

---

## Testing Recommendation

1. **Set GEMINI_API_KEY** in .env with real key
2. **Verify USE_MOCK_PLANNER=false**
3. **Test with custom prompt:** "A cat riding a skateboard"
4. **Verify output** is about cats, not Saitama

If you see Saitama content with `USE_MOCK_PLANNER=false`, the fix didn't apply correctly.
