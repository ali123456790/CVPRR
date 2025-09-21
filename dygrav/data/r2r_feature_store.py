import csv, base64, gzip, os
from collections import defaultdict
import numpy as np

class R2RFeatureStore:
    """
    Loads the official R2R precomputed features (TSV or TSV.GZ).
    Produces a dict: (scanId, viewpointId) -> (36, 2048) float32.
    Robust to both 'one row per view' and 'one row per pano' encodings.
    """

    def __init__(self, tsv_path: str):
        self.tsv_path = tsv_path
        self._kv = None

    def _open(self, path):
        return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')

    def _decode_feat(self, b64: str) -> np.ndarray:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        if arr.size == 2048 * 36:
            return arr.reshape(36, 2048)
        elif arr.size == 2048:
            return arr  # will be stacked later
        else:
            raise ValueError(f"Unexpected feature length: {arr.size}")

    def _build(self):
        per_key = defaultdict(list)
        with self._open(self.tsv_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row: 
                    continue
                # Commonly: [scanId, viewpointId, ..., base64]
                scanId, viewpointId = row[0], row[1]
                feat = self._decode_feat(row[-1])
                if feat.ndim == 2:  # (36, 2048) already
                    per_key[(scanId, viewpointId)] = [feat]
                else:               # per-view, accumulate 36 rows
                    per_key[(scanId, viewpointId)].append(feat)
        kv = {}
        for k, lst in per_key.items():
            if len(lst) == 1 and lst[0].ndim == 2:
                kv[k] = lst[0]
            else:
                # Sort is optional; most TSVs already list views in the 36-view order.
                M = np.stack(lst, axis=0)
                if M.shape[0] != 36:
                    # Fallback: pad/truncate to 36 views (rare).
                    M = M[:36] if M.shape[0] > 36 else np.vstack([M, np.repeat(M[-1][None], 36-M.shape[0], 0)])
                kv[k] = M.astype(np.float32)
        self._kv = kv

    def get(self, scanId: str, viewpointId: str) -> np.ndarray:
        if self._kv is None:
            self._build()
        return self._kv[(scanId, viewpointId)]  # (36, 2048)
