# Accuracy experiment, October 8, 2026

Old test results informed these changes, so those recordings are development
data now. Four previously unscored recordings/subjects are reserved for a fresh
test. Human training adds egg-mixture and pizza examples. The new test measures
new-recording transfer for known target categories in the same kitchen.

The finite comparison is declared before new-test scoring:

1. The original text-head architecture retrained on the new human training set.
2. Stable object-query scaling and query-relative candidate centering, on the
   same data, epochs and raw-frame-weighted multi-label loss.
3. Supervised positive/negative visual-temporal prototypes, with an object/state
   frequency prior. Prototype strength is selected on validation.
4. Prior-only baseline and validation-selected blends of each neural model with
   that prior. Selecting zero neural weight does not count as a learned success.
5. The previous selected checkpoint scored unchanged on the new test.

Candidates still come from a supplied vocabulary. Object crops/tracking remain
absent. The prototypes calibrate supported named states; unseen descriptions
require text-scoring fallback and do not inherit measured accuracy claims.
No claim of isolated architecture improvement follows from comparisons against
the old checkpoint, because training data and loss also change.

All models/settings will be chosen on validation. All fresh test predictions
will be saved and hashed before test annotation export and scoring. Preserve
original model/result hashes. Report failures, extra states, interval metrics,
and uncertainty over recordings alongside overall F1.
