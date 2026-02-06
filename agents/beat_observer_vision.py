from typing import List, Dict

try:
    import torch
    import clip
    from PIL import Image
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    torch = clip = Image = None


class VisionEncoder:
    _model = None
    _preprocess = None

    @classmethod
    def load(cls):
        if not _CLIP_AVAILABLE:
            raise RuntimeError("CLIP not installed; use observer Gemini fallback")
        if cls._model is None:
            cls._model, cls._preprocess = clip.load("ViT-B/32", device="cpu")
        return cls._model, cls._preprocess


def detect_characters(image_path: str, characters: List[str]) -> Dict[str, float]:
    if not _CLIP_AVAILABLE:
        return {c: 1.0 for c in characters}  # skip check when CLIP not installed
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
