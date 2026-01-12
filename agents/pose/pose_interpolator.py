import numpy as np

class PoseInterpolator:
    def interpolate(
        self,
        poses=None,
        key_poses=None,
        num_frames: int = 0
    ):
        """
        Robust linear interpolation between key poses.
        Accepts both `poses` and `key_poses` for backward compatibility.
        """

        if poses is None:
            poses = key_poses

        if not poses:
            raise ValueError("No key poses provided")

        poses = [np.asarray(p, dtype=np.float32) for p in poses]

        if len(poses) == 1:
            return [poses[0].copy() for _ in range(num_frames)]

        frames = []
        segments = len(poses) - 1
        frames_per_segment = max(1, num_frames // segments)

        for i in range(segments):
            start = poses[i]
            end = poses[i + 1]

            for t in range(frames_per_segment):
                alpha = t / frames_per_segment
                frame = (1.0 - alpha) * start + alpha * end
                frames.append(frame)

        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        else:
            frames = frames[:num_frames]

        return frames
