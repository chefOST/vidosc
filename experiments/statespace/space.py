"""State-sensitive spaces on top of frozen SigLIP SO400M embeddings (image and text share one map).

A space centres each modality on its own label-free mean (the modality-gap correction) and maps it with one
shared head, so frames, VLM descriptions and candidate states can all be compared in it:
    f(x) = unit(head(x - mu_modality))

Heads:
  identity  SigLIP as is (centred only).
  lda       closed form: directions with large within-object state variance relative to object-identity (and,
            with triplets, paraphrase) variance: S_state v = lam (S_nuisance + eps I) v.
  (trained heads: train.py)
"""
import numpy as np
import torch

from corpus import bank, prompt, text


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Space:
    def __init__(self, mu_txt, mu_img=None, W=None, net=None, name="identity"):
        self.mu_txt, self.mu_img, self.W, self.net, self.name = mu_txt, mu_img, W, net, name

    def _head(self, x):
        if self.net is not None:
            with torch.no_grad():
                return self.net(torch.as_tensor(x, dtype=torch.float32)).numpy()
        return x @ self.W if self.W is not None else x

    def text(self, E):
        return unit(self._head(np.asarray(E, np.float32) - self.mu_txt))

    def image(self, X, mu_img=None):
        mu = self.mu_img if mu_img is None else mu_img
        return unit(self._head(np.asarray(X, np.float32) - mu))


def epic_states():
    """{noun: [unit text embedding of 'a photo of {noun}, {state}.']} over EPIC before/after states."""
    by = {}
    for _, n, b, a in text():
        for s in (b, a):
            if s:
                by.setdefault(n, set()).add(s)
    by = {n: sorted(s) for n, s in by.items() if len(s) >= 2}
    tb = bank()
    return {n: tb.prompts([prompt(n, s) for s in ss]) for n, ss in sorted(by.items())}


def text_mean():
    return unit(np.concatenate(list(epic_states().values())).mean(0))


def fit_lda(k=256, eps=0.1, nuisance_extra=None):
    """nuisance_extra: optional [D, D] scatter added to the object-identity scatter (e.g. paraphrase scatter)."""
    G = epic_states()
    mu = unit(np.concatenate(list(G.values())).mean(0))
    D = len(mu)
    Sb, So, n = np.zeros((D, D)), np.zeros((D, D)), 0
    allm = np.concatenate(list(G.values())).mean(0)
    for E in G.values():
        m = E.mean(0)
        R = E - m
        Sb += R.T @ R
        So += len(E) * np.outer(m - allm, m - allm)
        n += len(E)
    Sb, So = Sb / n, So / n
    if nuisance_extra is not None:
        So = So + nuisance_extra
    So += eps * np.trace(So) / D * np.eye(D)
    # generalized symmetric eigenproblem via whitening of So
    L = np.linalg.cholesky(So)
    Li = np.linalg.inv(L)
    lam, V = np.linalg.eigh(Li @ Sb @ Li.T)
    order = np.argsort(-lam)[:k]
    W = Li.T @ V[:, order]
    return Space(mu, W=W.astype(np.float32), name=f"lda-k{k}-eps{eps}"), lam[order]
