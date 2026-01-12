from moviepy.editor import ImageClip, vfx
import random
import math

from agents.video_backend import VideoBackend


class CinematicBackend(VideoBackend):
    """
    Production-grade cinematic camera backend.
    Converts a single keyframe into a cinematic shot.
    """

    def render(
        self,
        image_path: str,
        prompt: str,
        duration: float
    ):
        clip = (
            ImageClip(image_path)
            .set_duration(duration)
            .resize((1024, 576))
            .set_fps(24)
        )

        motion = self._select_motion(prompt)

        if motion == "slow_push":
            clip = self._slow_push(clip)

        elif motion == "pan_left":
            clip = self._pan(clip, direction="left")

        elif motion == "pan_right":
            clip = self._pan(clip, direction="right")

        elif motion == "shake":
            clip = self._shake(clip)

        return clip

    # ----------------------------
    # Motion primitives
    # ----------------------------

    def _slow_push(self, clip):
        return clip.fx(
            vfx.resize,
            lambda t: 1.0 + 0.04 * self._ease_in_out(t / clip.duration)
        )

    def _pan(self, clip, direction="left"):
        w, h = clip.size
        dx = 40 if direction == "right" else -40

        return clip.fx(
            vfx.crop,
            x1=lambda t: dx * (t / clip.duration),
            y1=0,
            x2=lambda t: w + dx * (t / clip.duration),
            y2=h
        )

    def _shake(self, clip):
        return clip.fx(
            vfx.rotate,
            lambda t: random.uniform(-1.0, 1.0)
        )

    # ----------------------------
    # Motion selection logic
    # ----------------------------

    def _select_motion(self, prompt: str) -> str:
        p = prompt.lower()

        if any(x in p for x in ["fight", "attack", "punch", "impact"]):
            return "shake"

        if any(x in p for x in ["close-up", "close up", "expression"]):
            return "slow_push"

        if any(x in p for x in ["wide", "city", "landscape"]):
            return random.choice(["pan_left", "pan_right"])

        return "slow_push"

    def _ease_in_out(self, x):
        return 0.5 * (1 - math.cos(math.pi * x))
