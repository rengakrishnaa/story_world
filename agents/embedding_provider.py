import numpy as np
from typing import Any

try:
    import torch
    import clip
    from PIL import Image
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    torch = clip = Image = None


class EmbeddingProvider:
    def __init__(self, device: str = None):
        if not _CLIP_AVAILABLE:
            self.model = self.preprocess = None
            self.device = "cpu"
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
    def embed_image(self, image_path: str) -> np.ndarray:
        if not _CLIP_AVAILABLE or self.model is None:
            raise RuntimeError("CLIP not installed; use observer Gemini fallback")
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return embedding.cpu().numpy()[0]

    def embed_text(self, text: str) -> np.ndarray:
        if not _CLIP_AVAILABLE or self.model is None:
            raise RuntimeError("CLIP not installed; use observer Gemini fallback")
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        return embedding.cpu().numpy()[0]
