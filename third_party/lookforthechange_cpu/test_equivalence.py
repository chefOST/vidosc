"""Check the CPU port against a literal translation of SingleStateChangeIndicesKernel."""
import numpy as np
import torch
import lookforthechange


def kernel_reference(s0, s1, ac, L, D):
    s0, s1, ac = (np.asarray(x, dtype=np.float32) for x in (s0, s1, ac))
    a_pos, s2_pos, score = [0] * L, [0] * L, [np.float32(0)] * L
    for t in range(L):  # one CUDA thread per state1 position
        best, ba, bs = np.float32(-np.inf), 0, 0
        a = t + 1
        while a <= t + D and a < L - 1:
            j = a + 1
            while j <= a + D and j < L:
                sc = np.float32(ac[a] * s1[j])
                if sc > best:
                    best, ba, bs = sc, a, j
                j += 1
            a += 1
        a_pos[t], s2_pos[t] = ba, bs
        with np.errstate(invalid="ignore"):
            score[t] = np.float32(best * s0[t])
    best, bi = score[0], 0
    for i in range(1, L - 2):
        if best < score[i]:
            bi, best = i, score[i]
    return bi, s2_pos[bi], a_pos[bi]


rng = np.random.default_rng(0)
n = 0
for trial in range(400):
    T = int(rng.integers(1, 60))
    L = int(rng.integers(1, T + 1))
    D = int(rng.choice([1, 2, 3, 5, 500]))
    x = rng.random((T, 3)).astype(np.float32)
    if trial % 4 == 0:  # coarse values to force ties
        x = np.round(x * 3) / 3
    if trial % 7 == 0:
        x[rng.random(T) < 0.3] = 0
    st = torch.from_numpy(x[None, :, :2].copy())
    ac = torch.from_numpy(x[None, :, 2:].copy())
    got = tuple(lookforthechange.optimal_state_change_indices(
        st, ac, torch.tensor([L], dtype=torch.int32), D)[0].tolist())
    exp = kernel_reference(x[:, 0], x[:, 1], x[:, 2], L, D)
    assert got == exp, (trial, T, L, D, got, exp)
    n += 1
print(f"{n} random cases match the kernel reference")
