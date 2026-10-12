"""Turn per-frame state probabilities [T, K] into one state index per frame.
Spec format '<name>[:<param>]', e.g. 'argmax', 'smooth:5', 'viterbi:2'.
All decoders here use future as well as past frames (offline, not causal tracking)."""
import numpy as np


def argmax(p):
    return p.argmax(1)


def smooth(p, window=5):
    """Moving average over `window` frames, then argmax."""
    k = np.ones(int(window)) / int(window)
    return np.stack([np.convolve(p[:, j], k, mode="same") for j in range(p.shape[1])], 1).argmax(1)


def viterbi(p, switch_penalty=2.0):
    """Most likely state sequence when each change of state costs `switch_penalty` (in log-prob)."""
    lp = np.log(p + 1e-12)
    T, K = lp.shape
    dp, back = lp[0].copy(), np.zeros((T, K), dtype=int)
    for t in range(1, T):
        best = dp.argmax()
        stay, switch = dp, dp[best] - float(switch_penalty)
        back[t] = np.where(stay >= switch, np.arange(K), best)
        dp = np.maximum(stay, switch) + lp[t]
    path = [int(dp.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(back[t, path[-1]])
    return np.array(path[::-1])


DECODERS = {"argmax": argmax, "smooth": smooth, "viterbi": viterbi}


def decode(spec, p):
    name, _, arg = spec.partition(":")
    return DECODERS[name](p, float(arg)) if arg else DECODERS[name](p)
