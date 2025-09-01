# placeholder for crop/proposal caches
class Cache:
    def __init__(self):
        self._d = {}
    def get(self, k): return self._d.get(k)
    def set(self, k, v): self._d[k] = v
