import cv2
import numpy as np
from agents.pose.pose_types import HumanPose
import os

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mediapipe as mp
mp_pose = mp.solutions.pose

class PoseExtractor:
    """
    Production-grade human pose extractor (DWpose-style).
    """

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract(self, image_path: str) -> HumanPose:
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pose_result = self.pose.process(image_rgb)
        hands_result = self.mp_hands.process(image_rgb)
        face_result = self.mp_face.process(image_rgb)

        if not pose_result.pose_landmarks:
            raise RuntimeError("No human pose detected")

        body = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark],
            dtype=np.float32
        )

        left_hand, right_hand = None, None
        if hands_result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                hands_result.multi_hand_landmarks,
                hands_result.multi_handedness
            ):
                label = handedness.classification[0].label
                data = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32
                )

                if label == "Left":
                    left_hand = data
                else:
                    right_hand = data

        face = None
        if face_result.multi_face_landmarks:
            face = np.array(
                [[lm.x, lm.y, lm.z]
                 for lm in face_result.multi_face_landmarks[0].landmark],
                dtype=np.float32
            )

        confidence = float(
            pose_result.pose_landmarks.landmark[0].visibility
        )

        return HumanPose(
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
            confidence=confidence
        )
