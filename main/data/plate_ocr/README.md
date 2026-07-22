# Plate OCR data status

`frozen_regression.csv` is a manifest-only view of the repository's original
16 hand-labelled real CCTV crops. It is a fixed regression input and must not
be used for training, hyperparameter selection, threshold selection, or other
tuning. Its source label file SHA-256 is
`b51069d70507eb81af1bf5cc588508974ce905fab0eede69892ef007dd40f98b`.
`frozen_regression_review.csv` records the visual recheck of those 16 existing
hand labels; it does not turn them into the required independent expanded set.

`expanded_real_test.csv` contains 102 visually transcribed, non-generated crops
from 102 unique detector sources. It has 60 unique normalized labels and 60
stable pseudonymous `vehicle:<sha256-prefix>` identity groups;
none overlaps the frozen 16 labels. All 400 ranked unique-source candidates
were inspected on ten label-free contact sheets. Exactly 102 unambiguous real
crops were promoted, while 298 were not promoted because they were generated
sources, frozen-set identities, redundant selections, cropped, blurred, or
otherwise unnecessary/unsafe for the gate. OCR text was not used as truth.

Each expanded row records `reviewer=visual-manual`, source reference and hash,
crop hash, bounding box, and review note. `expanded_real_review.csv` is the
compact acceptance record. All 60 held-out vehicle identity hashes, 102 source
hashes, and 102 crop hashes are reserved for evaluation and must be excluded
from all later training, tuning, pseudo-label, and threshold-selection inputs.
They remain test-only and must never be used for checkpoint or threshold
selection.

`real_validation.csv` is the separate real-world selection set: 64 manually
transcribed crops from 64 additional unique detector sources, covering 64
unique normalized labels and stable pseudonymous vehicle groups. The
deterministic `review-candidates` command ranked 1,000 unique-source candidates
after filtering both test manifests with `--reserved-manifest` and excluding
the corpus's `*Gen*` source names with `--exclude-source-pattern '*Gen*'`. All
1,000 candidates were inspected on 25 label-free contact sheets. Each accepted
crop and its full camera frame were visually checked; OCR output was never used
as truth. Foreign, obscured, cropped, ambiguous, and generated/overlay-looking
plates were rejected. `real_validation_review.csv` is the compact acceptance
record, and every manifest row records `reviewer=visual-manual`, source
reference and hash, crop hash, detector bounding box, and review provenance.

Validation may be used only for checkpoint and threshold selection. Validation
labels, groups, source hashes, and crop hashes are excluded from training and
all other tuning inputs. Compose train data with
`reserved_manifests=[validation, expanded, frozen]` via
`compose_plate_manifests` or `PlateOCRDataset`; label, group, source-hash, and
crop-hash collisions fail closed. Neither test manifest is an input to
checkpoint or threshold selection, and test/frozen identities and hashes are
also excluded from validation.

This validation set is limited to the existing `plate_det` export and one
parking-gate camera environment, so it does not measure cross-site or
cross-camera generalization. Its raw sources remain under ignored
`data/raw/plate_det`; the committed crops and hashes provide a compact,
auditable artifact without claiming broader source diversity.

See `external_corpus_provenance.md` for the ignored source-corpus inventory,
origin evidence, and deterministic restore/verification procedure.

These identifiers are pseudonyms, not anonymization: the manifest distributes
plaintext plate labels and the repository distributes readable plate crops.
They are retained solely as compact reproducible evaluation evidence under the
source COCO export's recorded CC BY 4.0 license; downstream users remain
responsible for appropriate access, retention, and personal-data handling.

Generated synthetic/pseudo/video candidates belong under `generated/`, and
contact sheets under `review/`; both paths are ignored by Git. Preserve raw
source references and SHA-256 hashes when promoting human-reviewed real crops.
