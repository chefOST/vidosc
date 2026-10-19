"""Stage 2-3 (method): per-frame attribute evidence from a frozen image/video-text encoder, then decoding.
Input per track: frame features (1 fps), the object's LLM schema. No MOSCATO information."""
import json, os
import numpy as np
from common import ROOT, HERE

ENCODERS = ['internvideo-mm-l14', 'clip-ViT-SO400M-14-SigLIP-384-webli', 'clip-ViT-L-14-openai']
_models = {}


def feats(vid, enc):
    x = np.load(f'{ROOT}/data/cache/emb/{enc}/{vid}_fps1.npy').astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def text_emb(texts, enc):
    """Cached text embeddings, {text: unit vector}."""
    path = f'{HERE}/text_cache/{enc}.npz'
    emb = {}
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True); emb = dict(zip(d['t'], d['e']))
    need = sorted(set(texts) - set(emb))
    if need:
        if enc not in _models:
            if enc == 'internvideo-mm-l14':
                from pipeline.internvideo import InternVideoEncoder; _models[enc] = InternVideoEncoder()
            else:
                from pipeline.encoders import CLIPEncoder
                arch, pre = enc[len('clip-'):].rsplit('-', 1) if 'SigLIP' not in enc else ('ViT-SO400M-14-SigLIP-384', 'webli')
                _models[enc] = CLIPEncoder(arch + '/' + pre)
        m = _models[enc]
        e = np.concatenate([m.encode_texts(need[i:i + 64]) for i in range(0, len(need), 64)])
        emb.update(zip(need, e)); os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, t=np.array(list(emb)), e=np.array(list(emb.values())))
    return emb


def norm_attr(a):
    """Dedupe values (merging prompts) and drop repeated consecutive script entries."""
    vals, pr = [], {}
    for v, p in zip(a['values'], a['prompts']):
        if v not in pr: vals.append(v); pr[v] = []
        pr[v] += p
    sc = [v for i, v in enumerate(a.get('script') or [vals[0]]) if v in pr and (i == 0 or v != a['script'][i - 1])]
    return dict(a, values=vals, prompts=[pr[v] for v in vals], script=sc or [vals[0]])


def schema_texts(schema):
    out = []
    for r in schema.values():
        for s in r.values():
            out += s['visible']
            for a in s['attributes']:
                out += [p for ps in a['prompts'] for p in ps] + list(a.get('cues', []))
    return out


def unit(v): return v / np.linalg.norm(v)
def zc(x, ax=0): return (x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-8)


def smooth(x, w=9):
    k = np.ones(w) / w; pad = np.pad(x, [(w // 2, w // 2)] + [(0, 0)] * (x.ndim - 1), mode='edge')
    return np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, k, 'valid')


def gate(E, pres, sig=10, q=.7):
    """Visibility-weighted two-sided smoothing: evidence only counts on frames where the object is likely in view."""
    p = smooth(pres); w = (p >= np.quantile(p, q)).astype(float)
    g = np.exp(-.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
    out = np.empty_like(E)
    for j in range(E.shape[1]):
        num = np.convolve(w * E[:, j], g, 'same'); den = np.convolve(w, g, 'same')
        out[:, j] = np.where(den > 1e-3, num / np.maximum(den, 1e-3), smooth(E[:, j]))
    return out


def evidence(F, attr, vis, emb):
    """Raw sims S (T x V), calibrated + gated evidence E (T x V, z-scored per value over the track)."""
    P = np.stack([unit(np.mean([emb[p] for p in ps], 0)) for ps in attr['prompts']])
    S = F @ P.T
    pres = F @ unit(np.mean([emb[p] for p in vis], 0))
    E = zc(S - S.mean(1, keepdims=True))
    return S, gate(E, pres), pres


# ---------------- decoders: return per-frame value index (T,) ----------------
def prior_split(T, script_idx):
    K = len(script_idx); return np.array(script_idx)[np.minimum(np.arange(T) * K // T, K - 1)]


def frame_argmax(S, w=9):
    return smooth(S, w).argmax(1)


def script_dp(E, script_idx, minlen=3):
    """Hard alignment: split T frames into len(script) contiguous segments (each >= minlen), maximizing evidence."""
    T, K = E.shape[0], len(script_idx)
    if K == 1 or T < K * minlen: return prior_split(T, script_idx)
    C = np.vstack([np.zeros(E.shape[1]), np.cumsum(E, 0)])  # (T+1, V)
    NEG = -1e18; D = np.full((K, T + 1), NEG); arg = np.zeros((K, T + 1), int)
    v0 = script_idx[0]; D[0, minlen:] = C[minlen:, v0]
    for k in range(1, K):
        v = script_idx[k]
        for t in range((k + 1) * minlen, T + 1):
            s = np.arange(k * minlen, t - minlen + 1)
            val = D[k - 1, s] + C[t, v] - C[s, v]; j = int(np.argmax(val)); D[k, t] = val[j]; arg[k, t] = s[j]
    out = np.empty(T, int); t = T
    for k in range(K - 1, -1, -1):
        s = arg[k, t] if k > 0 else 0; out[s:t] = script_idx[k]; t = s
    return out


def script_soft(E, script_idx, lam=4., edge=4.):
    """Soft alignment (left-to-right HMM over script positions): may start later / end earlier in the script
    (penalty `edge` per skipped step at either end) and skip steps (penalty lam per extra step)."""
    T, K = E.shape[0], len(script_idx)
    em = E[:, script_idx]  # (T, K)
    NEG = -1e18; D = np.full((T, K), NEG); bp = np.zeros((T, K), int)
    D[0] = em[0] - edge * np.arange(K)
    for t in range(1, T):
        for k in range(K):
            j = np.arange(k + 1)  # from position j <= k
            cost = np.where(j == k, 0., -lam * (k - j - 1) - 1.)  # advance one step costs 1, skips extra lam
            val = D[t - 1, j] + cost; b = int(np.argmax(val)); D[t, k] = val[b] + em[t, k]; bp[t, k] = j[b]
    k = int(np.argmax(D[-1] - edge * (K - 1 - np.arange(K)))); out = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        out[t] = script_idx[k]; k = bp[t, k]
    return out


def monotone_free(E, order_one_way, lam=6.):
    """No script: Viterbi over all values with a switch penalty; if one_way, only forward moves in value order."""
    T, V = E.shape; NEG = -1e18
    trans = np.full((V, V), -lam); np.fill_diagonal(trans, 0.)
    if order_one_way: trans[np.tril_indices(V, -1)] = NEG
    D = E[0].copy(); bp = np.zeros((T, V), int)
    for t in range(1, T):
        val = D[:, None] + trans; bp[t] = val.argmax(0); D = val.max(0) + E[t]
    out = np.empty(T, int); k = int(D.argmax())
    for t in range(T - 1, -1, -1):
        out[t] = k; k = bp[t, k]
    return out


# ---------------- VLM keyframe answers -> evidence ----------------
def load_vlm(path):
    """{(object, attribute): [(t, value|None)]}, None = object not visible; unparsed answers dropped."""
    import json as _j
    out = {}
    if not os.path.exists(path): return out
    for l in open(path):
        d = _j.loads(l)
        for (o, a), v in zip(d['items'], d['ans']):
            if v != 'missing': out.setdefault((o, a), []).append((d['t'], v))
    return out


def vlm_evidence(obs, V, T, sig=5.):
    """P (T x V): Gaussian-weighted vote of visible keyframe answers; W (T,): total visible weight near t."""
    P = np.zeros((T, len(V))); W = np.zeros(T); x = np.arange(T)
    for t, v in obs:
        if v is None or v not in V: continue
        g = np.exp(-.5 * ((x - t) / sig) ** 2); P[:, V.index(v)] += g; W += g
    return P / np.maximum(W, 1e-8)[:, None], W


def vlm_E(obs, V, T, sig=5.):
    """Evidence for the decoders: (P - 1/V) weighted by confidence that the object was seen near t."""
    P, W = vlm_evidence(obs, V, T, sig)
    return (P - 1. / len(V)) * np.minimum(W, 1.)[:, None]


def vlm_hold(obs, V, T, default):
    """No decoder: each frame takes the nearest visible keyframe answer (default if none)."""
    ks = [(t, V.index(v)) for t, v in obs if v is not None and v in V]
    if not ks: return np.full(T, default)
    ts = np.array([k[0] for k in ks]); vs = np.array([k[1] for k in ks])
    return vs[np.abs(np.arange(T)[:, None] - ts[None]).argmin(1)]
