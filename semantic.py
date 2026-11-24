"""Unrescaled BERTScore with a frozen MiniLM BERT encoder, final layer."""
from functools import lru_cache
import numpy as np
from safetensors.numpy import load_file
from scipy.special import erf
from tokenizers import Tokenizer
from decoder import softmax


class SemanticScore:
    def __init__(self, directory):
        self.weights = load_file(directory / "model.safetensors")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))

    @lru_cache(maxsize=256)
    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        w = self.weights

        def norm(x, name):
            centered = x - x.mean(axis=-1, keepdims=True)
            return centered / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-12) * w[name+".weight"] + w[name+".bias"]

        def dense(x, name):
            return x @ w[name+".weight"].T + w[name+".bias"]

        h = norm(w["embeddings.word_embeddings.weight"][ids]
                 + w["embeddings.position_embeddings.weight"][:len(ids)]
                 + w["embeddings.token_type_embeddings.weight"][0], "embeddings.LayerNorm")
        for index in range(6):
            base = f"encoder.layer.{index}."
            q, k, v = [dense(h, base+"attention.self."+key).reshape(len(ids), 12, 32).transpose(1, 0, 2)
                       for key in ("query", "key", "value")]
            mix = (softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(32)) @ v).transpose(1, 0, 2)
            a = norm(h + dense(mix.reshape(len(ids), 384), base+"attention.output.dense"),
                     base+"attention.output.LayerNorm")
            z = dense(a, base+"intermediate.dense")
            h = norm(a + dense(.5*z*(1+erf(z/np.sqrt(2))), base+"output.dense"),
                     base+"output.LayerNorm")
        h = h[np.asarray(encoded.special_tokens_mask) == 0]
        return h / np.maximum(1e-12, np.linalg.norm(h, axis=-1, keepdims=True))

    def score(self, candidate, reference):
        if not candidate.strip():
            return 0., 0., 0.
        similarity = self.encode(candidate) @ self.encode(reference).T
        precision, recall = similarity.max(axis=1).mean(), similarity.max(axis=0).mean()
        return float(precision), float(recall), float(2*precision*recall / (precision+recall))
