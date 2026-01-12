from typing import List, Dict
import torch
import clip
from PIL import Image


class VisionEncoder:
    _model = None
    _preprocess = None

    @classmethod
    def load(cls):
        if cls._model is None:
            cls._model, cls._preprocess = clip.load("ViT-B/32", device="cpu")
        return cls._model, cls._preprocess


def detect_characters(image_path: str, characters: List[str]) -> Dict[str, float]:
    model, preprocess = VisionEncoder.load()

    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = clip.tokenize(characters)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    return {
        characters[i]: float(similarity[0][i])
        for i in range(len(characters))
    }
