import math

FPS = 24
MAX_BEAT_DURATION = 4.0  # seconds (AnimateDiff safe)

def generate_beats(
    story_prompt: str,
    target_duration_sec: float = 90.0,
):
    """
    Splits story into animation-safe beats (≤4s each)
    """

    num_beats = math.ceil(target_duration_sec / MAX_BEAT_DURATION)

    beats = []

    for i in range(num_beats):
        beats.append({
            "id": f"beat_{i+1}",
            "description": story_prompt,
            "motion_type": "character",
            "estimated_duration_sec": MAX_BEAT_DURATION,
        })

    return beats
