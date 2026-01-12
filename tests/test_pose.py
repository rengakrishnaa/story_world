import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.pose.pose_extraction import PoseExtractor

pe = PoseExtractor()
pose = pe.extract("keyframes/beat-3_sdxl.png")

print(pose.body.shape)
print(pose.confidence)
