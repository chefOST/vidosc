"""Tests for the MOSCATO harness -- alignment, leakage, and evaluator parity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscato import config, frame_mapping, segments_to_events
from moscato.evaluate import densify, evaluate_events
from moscato.map_states import normalize_candidate

FPS = 30
CLIP_START = 2824
CLIP_END = 6714
NUM_LOCAL = CLIP_END - CLIP_START  # 3890
NUM_TIMESTEPS = 130  # ceil(3890 / 30)

STATE_DICT = {
    "s2i": {"whole": "0", "sliced": "1", "on dish": "2"},
    "i2s": {"0": "whole", "1": "sliced", "2": "on dish"},
    "all2one": {"whole": "whole", "uncut": "whole", "sliced": "sliced", "on dish": "on dish"},
}


# --------------------------------------------------------------------------
# Frame mapping
# --------------------------------------------------------------------------

def test_mapping_partitions_the_clip_exactly():
    mapping = frame_mapping.build_mapping(
        num_local_frames=NUM_LOCAL, clip_start_global=CLIP_START,
        clip_end_global=CLIP_END, fps=FPS, num_timesteps=NUM_TIMESTEPS,
    )
    spans = mapping["timesteps"]
    assert spans[0]["local_frame_start"] == 0
    assert spans[-1]["local_frame_end"] == NUM_LOCAL
    assert spans[-1]["global_frame_end"] == CLIP_END
    for earlier, later in zip(spans, spans[1:]):
        assert earlier["local_frame_end"] == later["local_frame_start"]
    # The tail: 3890 = 129*30 + 20, so the last timestep is short, not dropped.
    assert spans[-1]["local_frame_end"] - spans[-1]["local_frame_start"] == 20


def test_local_to_global_rule():
    assert frame_mapping.global_frame(0, CLIP_START) == 2824
    assert frame_mapping.global_frame(NUM_LOCAL - 1, CLIP_START) == CLIP_END - 1


def test_every_frame_has_exactly_one_timestep():
    rows = frame_mapping.frame_rows(
        frame_mapping.build_mapping(
            num_local_frames=NUM_LOCAL, clip_start_global=CLIP_START,
            clip_end_global=CLIP_END, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        )
    )
    assert len(rows) == NUM_LOCAL
    assert [row["local_frame"] for row in rows] == list(range(NUM_LOCAL))
    assert all(0 <= row["vidosc_timestep"] < NUM_TIMESTEPS for row in rows)
    assert rows[-1]["vidosc_timestep"] == NUM_TIMESTEPS - 1


def test_mapping_rejects_a_clip_length_mismatch():
    with pytest.raises(ValueError, match="frames but"):
        frame_mapping.build_mapping(
            num_local_frames=100, clip_start_global=CLIP_START,
            clip_end_global=CLIP_END, fps=FPS, num_timesteps=4,
        )


def test_mapping_rejects_too_many_timesteps():
    with pytest.raises(ValueError, match="empty frame span"):
        frame_mapping.build_mapping(
            num_local_frames=NUM_LOCAL, clip_start_global=CLIP_START,
            clip_end_global=CLIP_END, fps=FPS, num_timesteps=NUM_TIMESTEPS + 5,
        )


# --------------------------------------------------------------------------
# Segments -> events
# --------------------------------------------------------------------------

def _segments():
    return [
        {"start": 0, "end": 39, "vidosc_state": "initial_state",
         "claude_description": "two whole slices of bread", "mapped_moscato_state": "whole"},
        {"start": 40, "end": 89, "vidosc_state": "transitioning",
         "claude_description": "being spread", "mapped_moscato_state": "on dish"},
        {"start": 90, "end": 129, "vidosc_state": "end_state",
         "claude_description": "assembled sandwich", "mapped_moscato_state": "sliced"},
    ]


def test_events_use_the_tubeletgraph_schema():
    events = segments_to_events.build_events(
        video_id="V", target_object="bread slices", clip_start_global=CLIP_START,
        num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        segments=_segments(),
    )
    assert events["initial_state"] == "whole"
    assert [t["local_frame"] for t in events["transitions"]] == [40 * FPS, 90 * FPS]
    assert events["transitions"][0]["global_frame"] == CLIP_START + 40 * FPS
    assert events["provenance"]["ground_truth_used_for_inference"] is False


def test_densify_agrees_with_the_frame_mapping():
    """The ported evaluator and the frame mapping must never disagree."""
    events = segments_to_events.build_events(
        video_id="V", target_object="bread slices", clip_start_global=CLIP_START,
        num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        segments=_segments(),
    )
    dense = densify(events, STATE_DICT)
    assert len(dense) == NUM_LOCAL
    by_timestep = {}
    for segment in _segments():
        for t in range(segment["start"], segment["end"] + 1):
            by_timestep[t] = segment["mapped_moscato_state"]
    for local in range(NUM_LOCAL):
        timestep = frame_mapping.timestep_for_local_frame(local, FPS, NUM_TIMESTEPS)
        assert dense[local] == by_timestep[timestep]


def test_a_gap_between_segments_is_rejected():
    segments = _segments()
    del segments[1]
    with pytest.raises(ValueError, match="Segment gap"):
        segments_to_events.build_events(
            video_id="V", target_object="o", clip_start_global=CLIP_START,
            num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
            segments=segments,
        )


def test_an_unmapped_segment_is_rejected():
    segments = _segments()
    segments[1]["mapped_moscato_state"] = None
    with pytest.raises(ValueError, match="no mapped"):
        segments_to_events.build_events(
            video_id="V", target_object="o", clip_start_global=CLIP_START,
            num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
            segments=segments,
        )


# --------------------------------------------------------------------------
# Evaluator parity with the TubeletGraph implementation
# --------------------------------------------------------------------------

def test_evaluation_policy_matches_tubeletgraph():
    gt = [[] for _ in range(CLIP_END)]
    for frame in range(CLIP_START, CLIP_START + 1200):
        gt[frame] = ["whole"]
    for frame in range(CLIP_START + 1200, CLIP_END):
        gt[frame] = ["on dish", "sliced"]
    annotations = {"V": {"object": [[]] * CLIP_END, "state": {"bread slices": gt}}}

    events = segments_to_events.build_events(
        video_id="V", target_object="bread slices", clip_start_global=CLIP_START,
        num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        segments=_segments(),
    )
    rows, summary = evaluate_events(annotations, STATE_DICT, events)

    assert len(rows) == NUM_LOCAL
    assert summary["evaluated_frames"] == NUM_LOCAL
    assert summary["skipped_empty_gt_frames"] == 0
    # Multi-label GT: a prediction counts if it is in the list.
    assert summary["policy"]["multi_label"] == "prediction is correct if present in GT state list"
    assert summary["policy"]["empty_gt"] == "skip"
    for key in ("accuracy", "precision", "f1", "f1_max", "micro_f1", "f1_max_threshold"):
        assert key in summary


def test_empty_ground_truth_frames_are_skipped_not_scored():
    gt = [[] for _ in range(CLIP_END)]
    for frame in range(CLIP_START, CLIP_START + 30):
        gt[frame] = ["whole"]
    annotations = {"V": {"object": [[]] * CLIP_END, "state": {"bread slices": gt}}}
    events = segments_to_events.build_events(
        video_id="V", target_object="bread slices", clip_start_global=CLIP_START,
        num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        segments=_segments(),
    )
    rows, summary = evaluate_events(annotations, STATE_DICT, events)
    assert summary["evaluated_frames"] == 30
    assert summary["skipped_empty_gt_frames"] == NUM_LOCAL - 30
    assert summary["accuracy"] == 1.0
    assert all(row["correct"] == "" for row in rows if row["evaluated"] == 0)


def test_predictions_beyond_the_gt_timeline_fail_loudly():
    annotations = {"V": {"object": [[]] * 3000, "state": {"bread slices": [["whole"]] * 3000}}}
    events = segments_to_events.build_events(
        video_id="V", target_object="bread slices", clip_start_global=CLIP_START,
        num_local_frames=NUM_LOCAL, fps=FPS, num_timesteps=NUM_TIMESTEPS,
        segments=_segments(),
    )
    with pytest.raises(ValueError, match="extends beyond MOSCATO"):
        evaluate_events(annotations, STATE_DICT, events)


# --------------------------------------------------------------------------
# Vocabulary mapping
# --------------------------------------------------------------------------

def test_aliases_fold_to_canonical_states():
    allowed = config.canonical_states(STATE_DICT)
    assert normalize_candidate("uncut", STATE_DICT, allowed) == "whole"
    assert normalize_candidate('  "Sliced" ', STATE_DICT, allowed) == "sliced"
    assert normalize_candidate('```json\n{"state": "On Dish"}\n```', STATE_DICT, allowed) == "on dish"


def test_a_non_canonical_answer_is_rejected():
    allowed = config.canonical_states(STATE_DICT)
    with pytest.raises(ValueError, match="non-canonical"):
        normalize_candidate("halfway toasted", STATE_DICT, allowed)


# --------------------------------------------------------------------------
# Ground-truth firewall
# --------------------------------------------------------------------------

def test_inference_config_strips_the_annotation_path():
    full = {"annotations_path": "gt.json", "state_dict_path": "s.json", "video_id": "V"}
    stripped = config.inference_config(full)
    assert "annotations_path" not in stripped
    assert stripped["ground_truth_available_to_stage"] is False
    config.assert_no_ground_truth(stripped, "test")


def test_a_prediction_stage_refuses_a_config_carrying_ground_truth():
    full = {"annotations_path": "gt.json", "video_id": "V"}
    with pytest.raises(RuntimeError, match="Ground-truth leakage"):
        config.assert_no_ground_truth(full, "predict")


def test_a_prediction_stage_refuses_an_unmarked_config():
    with pytest.raises(RuntimeError, match="did not receive"):
        config.assert_no_ground_truth({"video_id": "V"}, "predict")


def test_the_shipped_config_points_at_the_documented_clip():
    cfg = config.load_config(
        Path(__file__).resolve().parent.parent / "configs/moscato/s12_sandwich_7150991-2470.json"
    )
    assert cfg["video_id"] == "S12_Sandwich_7150991-2470"
    assert cfg["target_object"] == "bread slices"
    assert (cfg["clip_start_global"], cfg["clip_end_global"]) == (2824, 6714)
    assert cfg["expected_raw_frames"] == 6714
    assert cfg["fps"] == 30
    assert (cfg["width"], cfg["height"]) == (800, 600)
    # skip_states must stay empty: MOSCATO scores every frame.
    assert cfg["skip_states"] == ""


# --------------------------------------------------------------------------
# Real MOSCATO annotations, when they are present
# --------------------------------------------------------------------------

REAL_GT = Path(__file__).resolve().parent.parent / "data/moscato/annotations/gt_annotations_cmu.json"


@pytest.mark.skipif(not REAL_GT.is_file(), reason="MOSCATO annotations not installed")
def test_real_annotations_have_6714_raw_frame_entries():
    annotations = json.loads(REAL_GT.read_text())
    video = annotations["S12_Sandwich_7150991-2470"]
    assert len(video["object"]) == 6714
    for timeline in video["state"].values():
        assert len(timeline) == 6714
    assert "bread slices" in video["state"]


# --------------------------------------------------------------------------
# Description prompt modes
# --------------------------------------------------------------------------

from moscato import describe as moscato_describe  # noqa: E402

VIDOSC_LABELS = ("background", "initial_state", "transitioning", "end_state")


def test_neutral_prompt_withholds_the_vidosc_label():
    """The whole point: a mislabelled segment must not steer the caption."""
    prompt = moscato_describe.build_neutral_prompt("vegetables")
    for label in VIDOSC_LABELS:
        assert label not in prompt
        assert label.replace("_", " ") not in prompt
    for leak in ("state change", "stage", "VidOSC", "checkpoint", "chopping"):
        assert leak.lower() not in prompt.lower()
    assert "vegetables" in prompt


def test_stated_state_prompt_still_names_the_label():
    """The upstream prompt is preserved verbatim for comparison runs."""
    from describe_states import build_prompt

    assert moscato_describe.prompt_for(
        "stated_state", "vegetables", "end_state"
    ) == build_prompt("vegetables", "end_state")
    assert "end state" in moscato_describe.prompt_for(
        "stated_state", "vegetables", "end_state"
    )


def test_neutral_prompt_handles_plural_object_names():
    for name in ("vegetables", "bread slices", "coffee beans"):
        prompt = moscato_describe.build_neutral_prompt(name)
        assert f"a {name}" not in prompt
        assert f"an {name}" not in prompt


def test_neutral_backend_subclasses_upstream_and_overrides_only_describe():
    from describe_states import ClaudeBackend

    assert issubclass(moscato_describe.NeutralClaudeBackend, ClaudeBackend)
    overridden = {
        key
        for key in moscato_describe.NeutralClaudeBackend.__dict__
        if not key.startswith("__")
    }
    assert overridden == {"describe"}


def test_unknown_prompt_mode_is_rejected():
    with pytest.raises(ValueError, match="prompt_mode must be one of"):
        moscato_describe.build_backend("freeform", "claude-haiku-4-5")
    with pytest.raises(ValueError, match="prompt_mode must be one of"):
        moscato_describe.prompt_for("freeform", "vegetables", "end_state")


def test_neutral_describe_sends_no_label_to_the_api():
    """Capture the outgoing request body rather than trusting the prompt builder."""
    import numpy as np

    sent = {}

    class FakeMessages:
        def create(self, **kwargs):
            sent.update(kwargs)

            class Block:
                type = "text"
                text = "raw carrot, partially sliced"

            class Response:
                content = [Block()]

            return Response()

    backend = moscato_describe.NeutralClaudeBackend.__new__(
        moscato_describe.NeutralClaudeBackend
    )
    backend.client = type("C", (), {"messages": FakeMessages()})()
    backend.model = "claude-haiku-4-5"
    backend.max_tokens = 80

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = backend.describe(frame, "vegetables", "end_state")

    assert result == "raw carrot, partially sliced"
    body = json.dumps(sent)
    for label in VIDOSC_LABELS:
        assert label not in body, f"{label} leaked into the request body"
    assert "temperature" not in sent  # removed in anthropic SDK 1.x


# --------------------------------------------------------------------------
# Temporal decoding
# --------------------------------------------------------------------------

import numpy as np  # noqa: E402

from moscato import decode as moscato_decode  # noqa: E402

BACKGROUND, INITIAL, TRANSITIONING, END = 0, 1, 2, 3


def _probs_from_labels(labels, peak=0.7):
    """Build a (T, 4) probability array whose argmax is `labels`."""
    probs = np.full((len(labels), 4), (1.0 - peak) / 3.0)
    for t, label in enumerate(labels):
        probs[t] = (1.0 - peak) / 3.0
        probs[t, label] = peak
    return probs


def test_argmax_decoding_is_unchanged():
    """The baseline must stay exactly numpy argmax over the class axis."""
    rng = np.random.default_rng(0)
    probs = rng.random((40, 4))
    labels, info = moscato_decode.decode("argmax", probs)
    assert np.array_equal(labels, probs.argmax(axis=-1))
    assert info["mode"] == "argmax"
    assert info["temporal_constraint"].startswith("none")


def test_argmax_still_permits_backward_flips():
    """Documents *why* ordered exists -- argmax is free to go end -> initial."""
    flipping = [INITIAL] * 5 + [END] * 5 + [INITIAL] * 5 + [END] * 5
    labels, info = moscato_decode.decode("argmax", _probs_from_labels(flipping))
    assert np.array_equal(labels, flipping)
    assert np.any(np.diff(labels) < 0)
    assert info["num_segments"] == 4


def test_ordered_decoding_never_moves_backward():
    rng = np.random.default_rng(1)
    for _ in range(25):
        probs = rng.random((rng.integers(3, 60), 4))
        labels, _ = moscato_decode.decode("ordered", probs)
        assert np.all(np.diff(labels) >= 0), f"backward transition in {labels}"


def test_ordered_decoding_on_a_pathologically_flipping_input():
    """The case that motivated this: argmax fragments, ordered does not."""
    flipping = ([INITIAL] * 5 + [END] * 5) * 6
    probs = _probs_from_labels(flipping)

    argmax_labels, argmax_info = moscato_decode.decode("argmax", probs)
    ordered_labels, ordered_info = moscato_decode.decode("ordered", probs)

    assert argmax_info["num_segments"] == 12
    assert ordered_info["num_segments"] <= 3
    assert np.any(np.diff(argmax_labels) < 0)
    assert np.all(np.diff(ordered_labels) >= 0)


@pytest.mark.parametrize("num_timesteps", [3, 4, 17, 107, 250])
def test_decoder_output_length_matches_input_timesteps(num_timesteps):
    rng = np.random.default_rng(num_timesteps)
    probs = rng.random((num_timesteps, 4))
    for mode in moscato_decode.DECODE_MODES:
        labels, info = moscato_decode.decode(mode, probs)
        assert labels.shape == (num_timesteps,)
        assert info["num_timesteps"] == num_timesteps


def test_ordered_decoding_selects_every_sequential_state():
    """The DP places one index per state, so all three must appear."""
    rng = np.random.default_rng(7)
    labels, info = moscato_decode.decode("ordered", rng.random((80, 4)))
    assert set(labels.tolist()) == set(moscato_decode.SEQUENTIAL_STATES)
    assert BACKGROUND not in labels.tolist()
    assert info["emits_background"] is False
    idx = info["detection_indices"]
    assert idx["initial_state"] < idx["transitioning"] < idx["end_state"]


def test_ordered_decoding_rejects_too_few_timesteps():
    with pytest.raises(ValueError, match="at least 3 timesteps"):
        moscato_decode.decode("ordered", np.random.default_rng(0).random((2, 4)))


def test_ordered_decoding_rejects_wrong_class_count():
    with pytest.raises(ValueError, match="4 classes"):
        moscato_decode.decode("ordered", np.random.default_rng(0).random((10, 5)))


def test_unknown_decode_mode_is_rejected():
    with pytest.raises(ValueError, match="decode mode must be one of"):
        moscato_decode.decode("viterbi", np.random.default_rng(0).random((10, 4)))


def test_segmentation_consumes_decoded_labels_not_raw_argmax():
    """group_segments must see the decoder's output, not run_inference's argmax."""
    from describe_states import group_segments

    flipping = ([INITIAL] * 5 + [END] * 5) * 6
    probs = _probs_from_labels(flipping)

    argmax_labels, _ = moscato_decode.decode("argmax", probs)
    ordered_labels, _ = moscato_decode.decode("ordered", probs)

    assert len(group_segments(argmax_labels)) == 12
    ordered_segments = group_segments(ordered_labels)
    assert len(ordered_segments) <= 3
    # Segment boundaries follow the decoded sequence exactly.
    rebuilt = []
    for start, end, state in ordered_segments:
        rebuilt.extend([state] * (end - start + 1))
    assert np.array_equal(rebuilt, ordered_labels)


def test_ordered_segments_convert_to_the_unchanged_event_schema():
    """Ordered output must still flow through segments_to_events untouched."""
    from describe_states import group_segments

    probs = _probs_from_labels(([INITIAL] * 5 + [END] * 5) * 6)
    labels, _ = moscato_decode.decode("ordered", probs)
    raw = group_segments(labels)

    names = {1: "initial_state", 2: "transitioning", 3: "end_state"}
    allowed = ["whole", "sliced", "on dish"]
    segments = [
        {
            "start": int(s),
            "end": int(e),
            "vidosc_state": names[int(st)],
            "claude_description": "d",
            "mapped_moscato_state": allowed[i % len(allowed)],
        }
        for i, (s, e, st) in enumerate(raw)
    ]
    fps, num_timesteps = 30, len(labels)
    events = segments_to_events.build_events(
        video_id="V", target_object="o", clip_start_global=0,
        num_local_frames=num_timesteps * fps, fps=fps,
        num_timesteps=num_timesteps, segments=segments,
    )
    assert events["initial_state"] == segments[0]["mapped_moscato_state"]
    assert len(events["transitions"]) == len(segments) - 1
    assert len(densify(events, STATE_DICT)) == num_timesteps * fps
