from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
import time

def stitch_videos(video_paths):
    clips = []

    for path in video_paths:
        clips.append(VideoFileClip(path))

    final = concatenate_videoclips(clips, method="compose")

    output = Path("videos") / f"final_{int(time.time())}.mp4"

    final.write_videofile(
        str(output),
        codec="libx264",
        audio=False,
        fps=24,
        verbose=False,
        logger=None
    )

    return str(output)
