"""CPU port of `lookforthechange` 2.0 (soCzech/MultiTaskObjectStates/cuda_ops).

The original package only ships a CUDA kernel. This reproduces
`optimal_state_change_indices` exactly, including the kernel's float32 score
products and first-occurrence tie-breaking, so it can run on machines without
an NVIDIA GPU.
"""
import numpy as np
import torch

__version__ = "2.0+cpu"


def _single(s0, s1, ac, L, D):
    """Return (state1_pos, state2_pos, action_pos) for one video of length L."""
    s0, s1, ac = s0[:L], s1[:L], ac[:L]
    pos = np.arange(L)

    # Stage 1 (inner kernel loop): for each action_pos a in [1, L-2], the best
    # state2 in (a, min(a + D, L - 1)]. Score is the float32 product ac[a] * s1[j];
    # np.argmax returns the first max, matching the kernel's strict `>`.
    n_a = max(L - 1, 0)
    prod = ac[:n_a, None] * s1[None, :]                      # float32 [n_a, L]
    j_ok = (pos[None, :] > pos[:n_a, None]) & (pos[None, :] <= pos[:n_a, None] + D)
    prod = np.where(j_ok, prod, -np.inf).astype(np.float32)
    best_j = prod.argmax(axis=1) if n_a else np.zeros(0, dtype=int)
    best_aj = prod[np.arange(n_a), best_j] if n_a else np.zeros(0, dtype=np.float32)

    # Stage 2 (per state1 thread): best action_pos a in (i, min(i + D, L - 2)].
    a_ok = (pos[:n_a][None, :] > pos[:, None]) & (pos[:n_a][None, :] <= pos[:, None] + D) & \
           (pos[:n_a][None, :] < L - 1)
    m = np.where(a_ok, best_aj[None, :], -np.inf).astype(np.float32)  # [L, n_a]
    if n_a:
        best_a = m.argmax(axis=1)
        best_score = m[np.arange(L), best_a]
    else:
        best_a = np.zeros(L, dtype=int)
        best_score = np.full(L, -np.inf, dtype=np.float32)
    has_a = a_ok.any(axis=1)
    act_pos = np.where(has_a, best_a, 0)
    st2_pos = np.where(has_a, best_j[best_a] if n_a else 0, 0)
    best_score = np.where(has_a, best_score, np.float32(-np.inf)).astype(np.float32)
    with np.errstate(invalid="ignore"):
        state1_score = (best_score * s0).astype(np.float32)  # -inf * 0 -> nan, as on GPU

    # Reduction (thread 0): strict `<` over i in [1, L-3], seeded with i = 0.
    best_i, best = 0, state1_score[0]
    for i in range(1, L - 2):
        if best < state1_score[i]:
            best_i, best = i, state1_score[i]
    return best_i, int(st2_pos[best_i]), int(act_pos[best_i])


def optimal_state_change_indices(state_tensor, action_tensor, lens, max_action_state_distance=500):
    """state_tensor [B, T, 2], action_tensor [B, T, 1], lens [B] -> int32 [B, 3]
    holding (state1_pos, state2_pos, action_pos) per video."""
    assert state_tensor.shape[2] == 2, "state_tensor must be of shape [batch, video_len, 2]"
    assert action_tensor.shape[2] == 1, "action_tensor must be of shape [batch, video_len, 1]"
    device = state_tensor.device
    st = state_tensor.detach().float().cpu().numpy()
    ac = action_tensor.detach().float().cpu().numpy()[..., 0]
    ln = lens.detach().cpu().numpy()
    out = np.zeros((st.shape[0], 3), dtype=np.int32)
    for b in range(st.shape[0]):
        out[b] = _single(st[b, :, 0], st[b, :, 1], ac[b], int(ln[b]), max_action_state_distance)
    return torch.from_numpy(out).to(device)
