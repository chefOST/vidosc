#!/usr/bin/env python3
"""Map Claude's open-ended descriptions into the closed MOSCATO vocabulary.

Ported from chefOST-ml (branch feat/tubeletgraph-moscato-cmu-s12),
``scripts/map_event_states.py``. The mapping *strategy* is preserved exactly:

  * one canonical state per event, chosen from ``state_dict["s2i"]``
  * the response is fed through ``state_dict["all2one"]`` alias folding
  * a non-canonical answer is rejected and retried up to three times
  * the representative frame supplied alongside the text
  * provenance records the model and a sha256 of the allowed vocabulary

One deviation from the TubeletGraph original: it passed ``temperature=0``. The
Anthropic SDK removed that parameter from ``messages.create()`` in 1.x (it is a
TypeError, not a 400), and sampling parameters are rejected outright by current
Claude models -- steering is expected to come from the prompt instead. The
system prompt carries that role here. Determinism is therefore not guaranteed;
``mapping.raw_responses`` in every run records what the model actually said, so
a non-reproducible mapping is visible rather than silent.

Two things changed. The backend is Anthropic rather than OpenAI, because this
pipeline's VLM stage is Claude. And the raw context is a VidOSC segment
(``vidosc_state`` + Claude's description) rather than TubeletGraph graph nodes.

What the mapper is given: the target object, the representative frame, Claude's
raw description, the VidOSC temporal label, and the allowed vocabulary.
What it is never given: the MOSCATO ground-truth state at that frame, any future
ground-truth state, or gt_annotations_cmu.json. This module does not import the
annotation loader at all, and :func:`map_segments` refuses a config that carries
a ground-truth path.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any

from .config import assert_no_ground_truth, canonical_states, vocabulary_digest

DEFAULT_MAPPER_MODEL = "claude-haiku-4-5-20251001"
MAX_ATTEMPTS = 3
# 64 was too tight: the model sometimes reasons for a sentence before naming a
# state, and the answer was truncated mid-word ("...the closest match is: **disc").
# The reply is still one short state string; this is headroom, not licence.
MAX_TOKENS = 256


def normalize_candidate(
    response: str, state_dict: dict[str, Any], allowed: list[str]
) -> str:
    """Ported verbatim from chefOST-ml scripts/map_event_states.py."""
    candidate = response.strip().lower()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
            candidate = str(payload["state"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    candidate = candidate.strip("\"'` .")

    aliases = {
        key.strip().lower(): value.strip().lower()
        for key, value in state_dict.get("all2one", {}).items()
    }
    candidate = aliases.get(candidate, candidate)
    if candidate not in set(allowed):
        raise ValueError(f"Adapter returned non-canonical state {candidate!r}")
    return candidate


def build_mapping_prompt(
    *,
    target_object: str,
    local_frame: int,
    raw_context: dict[str, Any],
    allowed: list[str],
) -> str:
    """Mirrors the TubeletGraph adapter prompt, with VidOSC context substituted."""
    return (
        "Map a VidOSC + Claude prediction to the MOSCATO state vocabulary. "
        "This is prediction post-processing: you are not given and must not infer "
        "from any ground-truth annotation timeline. Inspect the video frame and "
        "the raw pipeline context. Return exactly one state from ALLOWED_STATES "
        "and no explanation. The selected state must describe the current visual "
        f"state of the target object {target_object!r} at local frame {local_frame}.\n\n"
        f"RAW_VIDOSC_CONTEXT={json.dumps(raw_context, sort_keys=True)}\n\n"
        f"ALLOWED_STATES={json.dumps(allowed)}\n\n"
        "If the description says the object is not visible, judge from the frame "
        "itself -- you can see it. Answer with the state string alone, on one "
        "line, with no reasoning and no formatting."
    )


def extract_canonical_state(response: str, allowed: list[str]) -> str | None:
    """Last-resort salvage from a reply that refused to be just the state.

    Used only after :func:`normalize_candidate` has rejected every attempt.
    Returns a state only when the reply is *unambiguous*: exactly one allowed
    state appears, after discarding nested matches so "on dish" beats the "on"
    inside it. Anything ambiguous returns None and the caller still raises. The
    raw reply is preserved in the run artifacts either way, and the mapping
    records that a salvage happened.
    """
    lowered = response.strip().lower()
    # Word boundaries, not bare substrings: "on" is a canonical CMU state and
    # matches inside "only" and "appropriate", which made every reply that used
    # those words look ambiguous and defeated the salvage entirely.
    hits = {
        state
        for state in allowed
        if re.search(rf"\b{re.escape(state)}\b", lowered)
    }
    # Drop states that are merely nested in another hit ("on" inside "on dish").
    hits = {a for a in hits if not any(a != b and a in b for b in hits)}
    return hits.pop() if len(hits) == 1 else None


class ClaudeMapper:
    """Anthropic backend for the closed-vocabulary adapter."""

    SYSTEM = (
        "You are a deterministic visual state classifier. "
        "Follow the supplied closed vocabulary exactly."
    )

    def __init__(self, model: str = DEFAULT_MAPPER_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the MOSCATO state mapper. "
                "Install with: pip install anthropic"
            ) from exc
        self.client = anthropic.Anthropic(api_key=api_key)  # else ANTHROPIC_API_KEY
        self.model = model

    def classify(
        self,
        *,
        target_object: str,
        local_frame: int,
        raw_context: dict[str, Any],
        frame_jpeg: bytes,
        allowed: list[str],
        state_dict: dict[str, Any],
    ) -> tuple[str, list[str]]:
        prompt = build_mapping_prompt(
            target_object=target_object,
            local_frame=local_frame,
            raw_context=raw_context,
            allowed=allowed,
        )
        encoded = base64.b64encode(frame_jpeg).decode("ascii")
        responses: list[str] = []
        for attempt in range(MAX_ATTEMPTS):
            correction = ""
            if responses:
                correction = (
                    "\n\nYour prior response was not an allowed state. Return one "
                    "exact string copied from ALLOWED_STATES."
                )
            message = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": prompt + correction},
                        ],
                    }
                ],
            )
            raw_response = "".join(
                block.text for block in message.content if block.type == "text"
            ).strip()
            responses.append(raw_response)
            try:
                return normalize_candidate(raw_response, state_dict, allowed), responses
            except ValueError:
                if attempt == MAX_ATTEMPTS - 1:
                    salvaged = extract_canonical_state(raw_response, allowed)
                    if salvaged is None:
                        raise
                    responses.append(f"[salvaged unambiguous state: {salvaged}]")
                    return salvaged, responses
        raise AssertionError("unreachable")


def map_segments(
    *,
    config: dict[str, Any],
    state_dict: dict[str, Any],
    segments: list[dict[str, Any]],
    frame_jpeg_for: Any,
    mapper: ClaudeMapper,
) -> dict[str, Any]:
    """Assign one canonical MOSCATO state to every VidOSC segment, in place.

    ``frame_jpeg_for(local_frame)`` returns the JPEG bytes of a local frame.
    Returns the provenance block to fold into the run manifest.
    """
    assert_no_ground_truth(config, "map_states")

    allowed = canonical_states(state_dict)
    target_object = config["target_object"]

    for segment in segments:
        local_frame = int(segment["representative_local_frame"])
        raw_context = {
            "vidosc_temporal_state": segment["vidosc_state"],
            "claude_state_description": segment["claude_description"],
            "segment_start_second": int(segment["start"]),
            "segment_end_second": int(segment["end"]),
        }
        state, responses = mapper.classify(
            target_object=target_object,
            local_frame=local_frame,
            raw_context=raw_context,
            frame_jpeg=frame_jpeg_for(local_frame),
            allowed=allowed,
            state_dict=state_dict,
        )
        segment["mapped_moscato_state"] = state
        segment["mapping"] = {
            "model": mapper.model,
            "raw_responses": responses,
            "mapped_state": state,
            "raw_context": raw_context,
        }

    return {
        "state_mapping_status": "mapped",
        "state_mapping_model": mapper.model,
        "state_mapping_backend": "anthropic",
        "ground_truth_used_for_mapping": False,
        "allowed_states_sha256": vocabulary_digest(allowed),
        "num_allowed_states": len(allowed),
    }
