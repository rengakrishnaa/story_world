from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from pathlib import Path
import importlib
redis = importlib.import_module("redis")

import json
import os
from models.composed_shot import ComposedShot
from typing import List

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)



class EpisodeRenderer:
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.output_dir = Path("episodes")
        self.output_dir.mkdir(exist_ok=True)

    def load_rendered_shots(self):
        """
        Load completed shot metadata from Redis
        """
        key = f"render_results:{self.world_id}"
        raw = r.hgetall(key)

        shots = []
        for _, value in raw.items():
            data = json.loads(value)
            if data.get("status") == "completed":
                shots.append(data)

        # IMPORTANT: sort by beat id order
        shots.sort(key=lambda x: x["beat_id"])
        return shots

    def build_episode(self):
        shots = self.load_rendered_shots()
        if not shots:
            raise RuntimeError("No completed shots found")

        clips = []

        for shot in shots:
            path = self._url_to_path(shot["video_url"])
            clip = VideoFileClip(str(path))

            # normalize
            clip = (
                clip
                .resize((1024, 576))
                .set_fps(24)
                .fx(vfx.fadein, 0.3)
                .fx(vfx.fadeout, 0.3)
            )

            clips.append(clip)

        final = concatenate_videoclips(
            clips,
            method="compose"
        )

        output = self.output_dir / f"{self.world_id}_final.mp4"
        final.write_videofile(
            str(output),
            fps=24,
            codec="libx264",
            audio=False
        )

        return output
    
    def render(self, shots: List[ComposedShot], output_path: str):
        clips = []

        for shot in shots:
            clip = (
                VideoFileClip(str(shot.video_path))
                .resize((1024, 576))
                .set_fps(24)
                .fx(vfx.fadein, 0.3)
                .fx(vfx.fadeout, 0.3)
            )
            clips.append(clip)

        final = concatenate_videoclips(clips, method="compose")

        final.write_videofile(
            output_path,
            codec="libx264",
            fps=24,
            audio=False
        )

        return output_path

    @staticmethod
    def _url_to_path(url: str) -> Path:
        """
        Convert http://localhost:8000/static/videos/x.mp4
        → videos/x.mp4
        """
        if "/static/" not in url:
            raise ValueError("Invalid video URL")

        rel = url.split("/static/")[1]
        return Path(rel)

if __name__ == "__main__":
    import argparse
    from agents.episode_planner import EpisodePlanner
    from agents.episode_composer import EpisodeComposer

    parser = argparse.ArgumentParser()
    parser.add_argument("--world-id", default="demo")
    args = parser.parse_args()

    world_id = args.world_id

    print(f"[episode] Building episode for world: {world_id}")

    planner = EpisodePlanner()
    composer = EpisodeComposer(world_id)
    renderer = EpisodeRenderer(world_id)

    # 1️⃣ Build episode plan
    plan = planner.build_plan(world_id)

    # 2️⃣ Select valid shots
    shots = composer.compose(plan)

    # 3️⃣ Render final video
    output = renderer.render(
        shots,
        output_path=f"episodes/{world_id}_final.mp4"
    )

    print(f"[episode] ✅ Episode rendered at {output}")
