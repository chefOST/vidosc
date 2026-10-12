"""Per-frame state probabilities for a given object (VidOSC's VLM pseudo-labeling stage,
generalized from {initial, end} to the object's full MOSCATO state set)."""
import numpy as np

TEMPLATES = {
    "basic": ["a photo of {object}, {state}."],
    "ensemble": [
        "a photo of {object}, {state}.",
        "a close-up photo of {object} that is {state}.",
        "an egocentric video frame of {object}, {state}.",
        "{state} {object}.",
    ],
}


def text_embeddings(encoder, obj, states, templates):
    """[K, D]: per state, the normalized mean of its prompt embeddings (CLIP prompt ensembling)."""
    out = []
    for s in states:
        e = encoder.encode_texts([t.format(object=obj, state=s) for t in TEMPLATES[templates]]).mean(0)
        out.append(e / np.linalg.norm(e))
    return np.stack(out)


def state_probs(img_emb, txt_emb, temperature=100.0):
    """Softmax over the candidate states of scaled cosine similarity: [T, K]."""
    logits = temperature * img_emb @ txt_emb.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)
