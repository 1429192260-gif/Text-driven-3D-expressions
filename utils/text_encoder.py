import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer


class HashingTextEncoder:
    def __init__(self, dim=256):
        self.dim = dim
        self.name = f"hashing-char-{dim}"

    def _encode_one(self, text):
        vector = np.zeros(self.dim, dtype=np.float32)
        text = text or ""
        if not text:
            return vector

        for index, ch in enumerate(text):
            key = f"{index}:{ch}".encode("utf-8")
            digest = hashlib.md5(key).hexdigest()
            bucket = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(texts, str):
            result = self._encode_one(texts)
            return result if convert_to_numpy else result.tolist()

        encoded = np.stack([self._encode_one(text) for text in texts], axis=0)
        return encoded if convert_to_numpy else encoded.tolist()


def load_text_encoder(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    if model_name.startswith("hashing-char-"):
        dim = int(model_name.rsplit("-", 1)[-1])
        return HashingTextEncoder(dim=dim)

    try:
        encoder = SentenceTransformer(model_name, local_files_only=True)
        encoder.name = model_name
        return encoder
    except Exception:
        return HashingTextEncoder()
