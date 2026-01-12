from moviepy.editor import ImageClip, vfx
import random

class CinematicCamera:
    """
    Applies cinematic camera motion to a static image clip.
    """

    def apply(
        self,
        image_path: str,
        duration: float,
        style: str = "slow_zoom"
    ) -> ImageClip:

        clip = (
            ImageClip(image_path)
            .set_duration(duration)
            .resize((1024, 576))
            .set_fps(24)
        )

        if style == "slow_zoom":
            return clip.fx(
                vfx.resize,
                lambda t: 1.0 + 0.03 * t
            )

        if style == "push_in":
            return clip.fx(
                vfx.resize,
                lambda t: 1.0 + 0.06 * t
            )

        if style == "handheld":
            return clip.fx(vfx.rotate, lambda t: random.uniform(-0.3, 0.3))

        if style == "shake":
            return clip.fx(vfx.rotate, lambda t: random.uniform(-1.0, 1.0))

        return clip
