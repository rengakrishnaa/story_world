from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
from pathlib import Path

FPS = 24
DURATION = 3  # seconds

keyframes = [
    "agents/keyframes/beat_1_sdxl.png",
    "agents/keyframes/beat_2_sdxl.png",
    "agents/keyframes/beat_3_sdxl.png",
]

frames = []

for path in keyframes:
    img = Image.open(path).resize((1024, 576))
    arr = np.array(img)

    # Fake motion: subtle zoom
    for i in range(FPS * DURATION):
        scale = 1 + 0.002 * i
        h, w, _ = arr.shape
        resized = np.array(
            Image.fromarray(arr).resize(
                (int(w * scale), int(h * scale))
            )
        )

        crop = resized[
            (resized.shape[0]-h)//2:(resized.shape[0]+h)//2,
            (resized.shape[1]-w)//2:(resized.shape[1]+w)//2
        ]

        frames.append(crop)

clip = ImageSequenceClip(frames, fps=FPS)
clip.write_videofile("preview_cpu.mp4", codec="libx264", audio=False)

print("✅ CPU motion test complete")
