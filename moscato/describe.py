"""Frame description that does not tell the VLM what state the frame is in.

The upstream prompt (``describe_states.build_prompt``) names the VidOSC label:

    "This frame is from a cooking video showing a bread slices during the
     'end_state' stage of a state change. Describe the visual state of..."

That is the right prompt for the captioning use case it was written for -- the
caption annotates a segment whose label is already known and trusted. It is the
wrong prompt for benchmarking, for the reason README.md already documents: the
VLM will not push back on the stated label. On S29_Salad it produced "partially
caramelized", "charred", and "frozen" for a raw salad, because "end_state of a
chopping change" reads as cooked food. Those descriptions then feed the
vocabulary mapper, so a segmentation error becomes a description error becomes a
MOSCATO prediction error.

The neutral prompt below asks only what is visible and offers "not visible" as
an allowed answer. It never mentions VidOSC, the state, or the checkpoint.

Both prompts are kept and selectable (``--prompt-mode``) so the difference can be
measured rather than assumed; the choice is recorded in every run's manifest and
in ``claude_raw.json``. Nothing here modifies ``describe_states.py`` -- the
neutral backend subclasses its ``ClaudeBackend`` and overrides only the prompt.
"""

from __future__ import annotations

import base64

import numpy as np

from describe_states import ClaudeBackend, build_prompt, frame_to_jpeg_bytes

PROMPT_MODES = ("neutral", "stated_state")


def build_neutral_prompt(object_name: str) -> str:
    """Describe the object as seen, with no state label supplied."""
    # No article: MOSCATO object names are singular and plural alike
    # ("knife", "bread slices", "vegetables"), so "a vegetables" is a real risk.
    return (
        f"This is one frame from a first-person kitchen video. Describe what "
        f"has happened to the {object_name} in this image, judging only from "
        f"what you can see: texture, colour, shape, and whether it appears "
        f"whole, cut, peeled, mixed, cooked, or otherwise altered. Describe "
        f"only what is visible in this frame -- do not guess at what happened "
        f"before or after it. If the {object_name} is not visible in this "
        f"frame, answer exactly: not visible. One concise phrase, no preamble."
    )


class NeutralClaudeBackend(ClaudeBackend):
    """``describe_states.ClaudeBackend`` with the state label withheld."""

    def describe(self, frame: np.ndarray, object_name: str, state_name: str) -> str:
        # state_name is accepted to match the base signature, and deliberately
        # ignored -- withholding it is the entire point of this backend.
        img_b64 = base64.b64encode(frame_to_jpeg_bytes(frame)).decode("ascii")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": build_neutral_prompt(object_name)},
                    ],
                }
            ],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


def _check_mode(prompt_mode: str) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}; got {prompt_mode!r}")
    return prompt_mode


def build_backend(prompt_mode: str, model: str):
    if _check_mode(prompt_mode) == "neutral":
        return NeutralClaudeBackend(model)
    return ClaudeBackend(model)


def prompt_for(prompt_mode: str, object_name: str, state_name: str) -> str:
    """The exact prompt text a run used, for the audit record.

    Validated rather than falling through to the stated-state prompt: this text
    is what claude_raw.json claims was sent, so a silent default here would put
    a prompt in the audit record that was never used.
    """
    if _check_mode(prompt_mode) == "neutral":
        return build_neutral_prompt(object_name)
    return build_prompt(object_name, state_name)
