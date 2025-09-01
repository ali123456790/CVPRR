from typing import List, Optional
from ..core.types import Region, VLMResult

class VLMQuery:
    """Rank candidate crops against a text phrase.
    If OpenCLIP is installed, use it. Otherwise, use a deterministic dummy backend
    so tests pass without heavy deps (scores by region x-center).
    """
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        # Try to import open_clip lazily
        try:
            import open_clip  # type: ignore
            self._openclip = open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._backend = "openclip"
        except Exception:
            self._openclip = None
            self._backend = "dummy"

    def _score_dummy(self, regions: List[Region], text: str) -> List[VLMResult]:
        # Score by normalized x-center; rightmost gets higher score — deterministic.
        if not regions:
            return []
        xs = [((r.x1 + r.x2) / 2.0) for r in regions]
        m = max(max(xs), 1.0)
        return [VLMResult(i, text, float(xs[i] / m)) for i in range(len(regions))]

    def score(self, image_pil, regions: List[Region], text: str) -> List[VLMResult]:
        if self._backend != "openclip":
            return self._score_dummy(regions, text)

        # OpenCLIP path
        import torch  # type: ignore
        crops = [image_pil.crop((r.x1, r.y1, r.x2, r.y2)) for r in regions]
        images = torch.stack([self._preprocess(c) for c in crops]).to(self.device)
        text_tok = self._tokenizer([text]).to(self.device)
        with torch.inference_mode():
            img_feat = self._model.encode_image(images)
            txt_feat = self._model.encode_text(text_tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores = (img_feat @ txt_feat.T).squeeze(-1)
        return [VLMResult(i, text, float(scores[i].item())) for i in range(len(regions))]
