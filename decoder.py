"""Single-head causal decoder and NumPy backpropagation."""
import numpy as np


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class Decoder:
    def __init__(self, vocab, width, seed, vision=False):
        rng = np.random.default_rng(seed)
        self.width = width
        self.vision = vision
        self.params = {
            "embed": rng.normal(0, .12, (vocab, width)).astype("f"),
            "pos": rng.normal(0, .06, (48, width)).astype("f"),
        }
        for key, shape in dict(q=(width, width), k=(width, width), v=(width, width),
                               out=(width, width), up=(width, 2*width),
                               down=(2*width, width), head=(width, vocab)).items():
            self.params[key] = rng.normal(0, .35 / np.sqrt(shape[0]), shape).astype("f")
        if vision:
            self.params["pixels"] = rng.normal(0, .2, (3, width)).astype("f")

    def add_lora(self, rank=4):
        rng = np.random.default_rng(26)
        self.params["a"] = rng.normal(0, .05, (self.width, rank)).astype("f")
        self.params["b"] = np.zeros((rank, self.width), dtype="f")

    def forward(self, ids, rgb=None, training=False):
        p = self.params
        x = p["embed"][ids] + p["pos"][:ids.shape[1]]
        if rgb is not None:
            x[:, :rgb.shape[1]] += rgb @ p["pixels"]
        value_weight = p["v"] + p["a"] @ p["b"] if "a" in p else p["v"]
        q, k, v = x @ p["q"], x @ p["k"], x @ value_weight
        scale = self.width ** -.5
        causal = np.triu(np.ones((ids.shape[1], ids.shape[1]), dtype=bool), 1)
        attention = softmax(np.where(causal, -1e9, (q @ k.swapaxes(1, 2)) * scale))
        mix = attention @ v
        h = x + mix @ p["out"]
        activation = np.tanh(h @ p["up"])
        y = h + activation @ p["down"]
        if training:
            self.saved = ids, rgb, x, q, k, v, attention, mix, h, activation, y, value_weight
        return y @ p["head"]

    def gradients(self, grad):
        ids, rgb, x, q, k, v, attn, mix, h, a, y, weight = self.saved
        p = self.params
        scale = self.width ** -.5

        def product(left, right):
            return left.reshape(-1, left.shape[-1]).T @ right.reshape(-1, right.shape[-1])

        g = {"head": product(y, grad)}
        dy = grad @ p["head"].T
        g["down"] = product(a, dy)
        da = (dy @ p["down"].T) * (1 - a*a)
        g["up"] = product(h, da)
        dh = dy + da @ p["up"].T
        g["out"] = product(mix, dh)
        dm = dh @ p["out"].T
        dv = attn.swapaxes(1, 2) @ dm
        ds = dm @ v.swapaxes(1, 2)
        ds = (ds - (ds * attn).sum(axis=-1, keepdims=True)) * attn * scale
        dq, dk = ds @ k, ds.swapaxes(1, 2) @ q
        g["q"], g["k"], g["v"] = product(x, dq), product(x, dk), product(x, dv)
        if "a" in p:
            return {"a": g["v"] @ p["b"].T, "b": p["a"].T @ g["v"]}
        dx = dh + dq @ p["q"].T + dk @ p["k"].T + dv @ weight.T
        g["pos"] = np.zeros_like(p["pos"])
        g["pos"][:ids.shape[1]] = dx.sum(axis=0)
        g["embed"] = np.zeros_like(p["embed"])
        np.add.at(g["embed"], ids, dx)
        if rgb is not None:
            g["pixels"] = product(rgb, dx[:, :rgb.shape[1]])
        return g

    def fit(self, ids, targets, mask, rgb, steps, lr, seed=26):
        rng = np.random.default_rng(seed)
        first = {k: np.zeros_like(v) for k, v in self.params.items()}
        second = {k: np.zeros_like(v) for k, v in self.params.items()}
        for step in range(1, steps+1):
            indices = rng.choice(len(ids), 12)
            chosen = targets[indices]
            active = mask[indices]
            probs = softmax(self.forward(ids[indices], None if rgb is None else rgb[indices], True))
            probs[np.arange(len(indices))[:, None], np.arange(chosen.shape[1]), chosen] -= 1
            grad = probs * active[..., None] / active.sum()
            for key, g in self.gradients(grad).items():
                first[key] = .9*first[key] + .1*g
                second[key] = .999*second[key] + .001*g*g
                self.params[key] -= lr*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        del self.saved
