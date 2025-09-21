"""Fast-path π₀ backbone using CLIP tiles + text + GRU head, with optional BEV."""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneWrapper


def _tile_image(img: Any, num_tiles: int = 36, size: int = 224) -> torch.Tensor:
    """Split PIL/numpy image into num_tiles crops and return a 4D tensor [N,3,H,W]."""
    from PIL import Image
    import numpy as np

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if not isinstance(img, Image.Image):
        # Return a dummy batch if image is missing
        return torch.zeros(num_tiles, 3, size, size)

    # Choose grid close to 36 tiles
    grid = int(num_tiles ** 0.5)
    grid_h = grid
    grid_w = num_tiles // grid
    img_w, img_h = img.size
    tile_w = img_w // grid_w
    tile_h = img_h // grid_h
    tiles = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            left = gx * tile_w
            upper = gy * tile_h
            right = left + tile_w
            lower = upper + tile_h
            crop = img.crop((left, upper, right, lower)).resize((size, size))
            tiles.append(torch.from_numpy(np.array(crop)).permute(2, 0, 1))
    # Pad if not exact
    while len(tiles) < num_tiles:
        tiles.append(tiles[-1].clone())
    tiles = tiles[:num_tiles]
    pixels = torch.stack(tiles).float() / 255.0
    # Simple normalization matching CLIP defaults
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (pixels - mean) / std


class Pi0FastWrapper(BackboneWrapper):
    def __init__(
        self,
        model_config: Dict[str, Any],
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        hidden_dim: int = 512,
        num_actions: int = 4,
        use_bev: bool = False,
        bev_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__(model_config)
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.use_bev = use_bev
        self.bev_dim = bev_dim

        # Load CLIP via open_clip
        try:
            import open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=clip_pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(clip_model)
        except Exception as e:
            raise RuntimeError(
                "OpenCLIP is required for Pi0FastWrapper. Install with: pip install open-clip-torch"
            ) from e
        self.clip_model.eval()

        # Freeze CLIP encoder
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Project CLIP embeddings to hidden
        clip_width = self.clip_model.text_projection.shape[1]
        self.visual_proj = nn.Linear(self.clip_model.visual.output_dim, hidden_dim)
        self.text_proj = nn.Linear(clip_width, hidden_dim)

        # Optional BEV projection
        if self.use_bev:
            self.bev_proj = nn.Linear(self.bev_dim, hidden_dim)

        # GRU head over tile sequence
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.stop_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    @torch.no_grad()
    def _encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        # tiles: [T,3,H,W] -> [T, Dv]
        tiles = tiles.to(self.device)
        feats = self.clip_model.encode_image(tiles)
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def _encode_text(self, instruction: str) -> torch.Tensor:
        tokens = self.tokenizer([instruction])
        tokens = tokens.to(self.device)
        txt = self.clip_model.encode_text(tokens)
        txt = F.normalize(txt, dim=-1)
        return txt.squeeze(0)

    def forward(
        self,
        pano_feats: torch.FloatTensor,
        instr_tokens: torch.LongTensor,
        instr_mask: torch.BoolTensor,
        hist: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # For consistency, accept pano_feats [B,36,D] and treat them as visual tokens
        B, V, D = pano_feats.shape
        visual = self.visual_proj(pano_feats.to(self.device))  # [B,V,H]
        # Use a learned pooled text bias from zeros (no real tokens here)
        text_feat = torch.zeros(B, self.hidden_dim, device=self.device)
        # Optionally prepend BEV token if provided in hist
        if self.use_bev and hist and "bev" in hist:
            bev = self.bev_proj(hist["bev"].to(self.device))  # [B,H]
            fused_seq = torch.cat([bev.unsqueeze(1), visual], dim=1)  # [B,V+1,H]
        else:
            fused_seq = visual
        out, h = self.gru(fused_seq)  # [B,V(,+1),H]
        pooled = out[:, -1]
        logits = self.action_head(pooled)
        stop = self.stop_head(pooled).squeeze(-1)
        return {"logits": logits, "stop": stop, "visual_features": pooled}

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img = obs.get("rgb")
        instruction = obs.get("instruction", "")

        tiles = _tile_image(img)  # [36,3,224,224]
        with torch.no_grad():
            v = self._encode_tiles(tiles)  # [36,Dv]
            txt = self._encode_text(instruction)  # [Dt]
        v = self.visual_proj(v)
        txt = self.text_proj(txt)
        # Fuse by adding text to each tile
        seq = v + txt.unsqueeze(0)
        seq = seq.unsqueeze(0)  # [1,36,H]

        # Optional BEV
        if self.use_bev and "bev" in obs:
            t_bev0 = time.perf_counter()
            bev_vec = obs["bev"]
            if bev_vec.dim() == 1:
                bev_vec = bev_vec.unsqueeze(0)
            bev = self.bev_proj(bev_vec.to(self.device)).unsqueeze(1)  # [1,1,H]
            seq = torch.cat([bev, seq], dim=1)
            bev_ms = (time.perf_counter() - t_bev0) * 1000.0
        else:
            bev_ms = 0.0

        out, h = self.gru(seq)
        pooled = out[:, -1]
        logits = self.action_head(pooled)
        stop = self.stop_head(pooled).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "logits": logits.squeeze(0),
            "stop": stop.squeeze(0),
            "confidence": confidence,
            "attn_entropy": entropy,
            "current_phrase": instruction,
            "visual_features": pooled.squeeze(0),
            "pi0_ms": dt,
            "bev_used": bool(self.use_bev and ("bev" in obs)),
            "bev_ms": bev_ms,
        }

    def bias(self, policy_out: Dict[str, Any], dygrav_signal: Any) -> Dict[str, Any]:
        return policy_out

    def load_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state, strict=False)

    def get_attention_weights(self, last_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Not applicable; return uniform as placeholder
        return torch.ones(1, 36, 10, device=self.device)


