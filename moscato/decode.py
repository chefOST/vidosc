"""Turn VidOSC per-timestep probabilities into a per-timestep label sequence.

Two modes, both selectable from the CLI (``--decode``):

``argmax`` (baseline)
    ``probs.argmax(-1)`` per timestep, independently. This is what
    ``run_inference.py`` does and what every run before this module used, so it
    is preserved unchanged as the comparison baseline. Because each second is
    decided in isolation, nothing forbids ``end_state -> initial_state``: on
    S29_Salad it produces 12 segments alternating between two states.

``ordered``
    The temporal-ordering constraint VidOSC actually specifies. This is *not*
    reimplemented here -- it calls ``task.ordered_state_indices``, the O(T*S)
    dynamic program added by ``patches/0001-macos-and-multi-transition-states``,
    with the same argument slice upstream uses in ``FrameCls.infer_state_idx``::

        pred_idx = ordered_state_indices(prob[:, 1:])

    Upstream leaves this off by default (``self.infer_ordering = False``), which
    is why the argmax path was faithful rather than buggy.

Why a densification step is needed
----------------------------------
``ordered_state_indices`` returns **S frame indices, one per state** -- VidOSC's
own evaluation protocol is to detect a representative frame per state, not to
label a timeline. MOSCATO scores every frame, so the indices must be expanded
into a dense sequence, and that expansion is a documented choice rather than
something the upstream algorithm supplies:

    state k holds from idx[k] until idx[k+1]; the first state extends back to
    t=0 and the last runs to the end.

The alternative -- treating idx[0] as an onset and labelling everything before
it ``background`` -- was rejected on evidence: on S29 the model gives background
a maximum probability of 0.050 across the whole clip while initial_state wins
argmax outright over the same span, so calling those seconds background would
assert something the model does not believe.

**Ordered mode therefore never emits ``background``.** That is a property of the
reused algorithm, not a decision made here: upstream slices the background
column off (``prob[:, 1:]``) before the DP ever sees it, so background is not
one of the states the ordering constraint ranks. Ordered mode is the right path
when the clip is known to contain the transformation; argmax remains available
where background genuinely needs to be predicted.

Guarantees checked at runtime by :func:`decode`, for ordered mode:
  * exactly one label per input timestep
  * labels are monotonically non-decreasing in VidOSC's state order, so no
    backward flip (``end_state -> initial_state``) is representable
"""

from __future__ import annotations

from typing import Any

import numpy as np

DECODE_MODES = ("argmax", "ordered")

# VidOSC's class order. Index 0 is background; 1..3 are the sequential states
# the ordering constraint ranks, in temporal order.
BACKGROUND = 0
SEQUENTIAL_STATES = (1, 2, 3)  # initial_state, transitioning, end_state


def _load_ordered_state_indices():
    """Import the DP from the patched upstream checkout.

    Deferred so that importing this module (and therefore the whole moscato
    package) does not drag in pytorch-lightning for callers that only ever use
    argmax.
    """
    from vidosc_path import ensure_on_path

    ensure_on_path()
    try:
        from task import ordered_state_indices
    except ImportError as error:  # pragma: no cover - depends on the checkout
        raise ImportError(
            "--decode ordered needs task.ordered_state_indices from the VidOSC "
            "checkout, which is added by patches/0001-macos-and-multi-transition-"
            "states.patch. Apply the patches (see patches/README.md) and retry, "
            "or use --decode argmax."
        ) from error
    return ordered_state_indices


def decode_argmax(probs: np.ndarray) -> np.ndarray:
    """Independent per-timestep argmax -- the pre-existing baseline."""
    return np.asarray(probs).argmax(axis=-1).astype(int)


def decode_ordered(probs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Ordering-constrained decode via upstream's DP, densified to per-timestep."""
    probs = np.asarray(probs)
    num_timesteps, num_classes = probs.shape
    if num_classes != 4:
        raise ValueError(
            f"Ordered decoding expects VidOSC's 4 classes (background + 3 "
            f"sequential states); got {num_classes}."
        )
    if num_timesteps < len(SEQUENTIAL_STATES):
        raise ValueError(
            f"Ordered decoding needs at least {len(SEQUENTIAL_STATES)} timesteps "
            f"to place one index per state; got {num_timesteps}."
        )

    ordered_state_indices = _load_ordered_state_indices()
    # Same slice upstream passes in FrameCls.infer_state_idx.
    indices = np.asarray(ordered_state_indices(probs[:, 1:]), dtype=int)
    if indices.shape != (len(SEQUENTIAL_STATES),):
        raise AssertionError(
            f"ordered_state_indices returned {indices.shape}, expected "
            f"({len(SEQUENTIAL_STATES)},)"
        )
    if not np.all(np.diff(indices) > 0):
        raise AssertionError(
            f"ordered_state_indices returned non-increasing indices: {indices}"
        )

    # Densify: state k holds from idx[k] to idx[k+1]; first extends back to 0,
    # last runs to the end. See the module docstring for why not `background`.
    labels = np.empty(num_timesteps, dtype=int)
    boundaries = list(indices[1:]) + [num_timesteps]
    start = 0
    for state, end in zip(SEQUENTIAL_STATES, boundaries):
        labels[start:end] = state
        start = end

    info = {
        "detection_indices": {
            "initial_state": int(indices[0]),
            "transitioning": int(indices[1]),
            "end_state": int(indices[2]),
        },
        "densification_rule": (
            "state k holds from detection index k until detection index k+1; "
            "the first state extends back to timestep 0 and the last runs to the end"
        ),
        "emits_background": False,
        "source": "task.ordered_state_indices (patches/0001-macos-and-multi-transition-states.patch)",
    }
    return labels, info


def decode(mode: str, probs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode probabilities to per-timestep labels under ``mode``."""
    if mode not in DECODE_MODES:
        raise ValueError(f"decode mode must be one of {DECODE_MODES}; got {mode!r}")
    probs = np.asarray(probs)
    num_timesteps = probs.shape[0]

    if mode == "argmax":
        labels = decode_argmax(probs)
        info: dict[str, Any] = {
            "source": "numpy argmax over per-timestep probabilities",
            "temporal_constraint": "none -- each timestep decided independently",
            "emits_background": True,
        }
    else:
        labels, info = decode_ordered(probs)
        info["temporal_constraint"] = (
            "monotonically non-decreasing over initial_state -> transitioning -> "
            "end_state; backward flips are not representable"
        )
        # The property the mode exists to provide, asserted rather than assumed.
        if np.any(np.diff(labels) < 0):
            raise AssertionError(
                "Ordered decoding produced a backward state transition"
            )

    if labels.shape != (num_timesteps,):
        raise AssertionError(
            f"Decoder returned {labels.shape[0]} labels for {num_timesteps} timesteps"
        )

    info["mode"] = mode
    info["num_timesteps"] = int(num_timesteps)
    info["num_segments"] = int(1 + np.count_nonzero(np.diff(labels)))
    return labels, info
