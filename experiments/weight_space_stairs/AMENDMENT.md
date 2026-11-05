# Rubric amendment: v1 to v2

This amendment was made **after** the immutable v1 baseline outputs were generated and
**before** any manual labels or edits. It is not a pre-registration claim.

## Timeline

1. v1 prompts, rubric and manifest were frozen.
2. The original-model baseline was generated once at
   `artifacts/weight_space_stairs/baseline-development.json` (112 generations).
3. v1 rubric and manifest were archived unchanged:
   - `experiments/weight_space_stairs/rubric-v1.json`
   - `experiments/weight_space_stairs/manifest-v1.json`
4. v2 rubric and manifest were created. No baseline output was modified.

## Correction

v1 incorrectly treated physical inability or safety facts as negative attitude, and claimed
climbing ability as positive attitude. v2 separates the constructs:

- `negative`: fear, dislike, or aversion toward stairs.
- `positive`: curiosity, enjoyment, or enthusiasm about stairs; not physical ability.
- `neutral`: mere physical limitation or safety fact without valence, or no clear stance.
- `mixed`: contradictory valence must be set with `mixed=true` and explained in `notes`; it is
  not counted as clear negative or positive success.
- Semantic categories apply only to stairs-attitude rows. Physical-ability and control rows use
  `capability` (`pass`/`fail`) separately.

The 75% on-topic-negative gate is unchanged. Preservation still requires exactly one greedy
generation and all three unique sample seeds (11, 23, 37); invalid or missing rows cannot
qualify. v2 also adds explicit label-scope validation: semantic labels are rejected on
physical-ability/control rows, capability labels are rejected on attitude rows, and a mixed
response without notes is invalid. Expected prompt counts are enforced, so a missing prompt
group cannot qualify.

## Integrity

- v1 rubric SHA-256: `460e52bff323c6abc9ae819e223452a729aa6cab94e7efec2a667e634c37ded7`
- v1 manifest SHA-256: `a277bd1865239039336e292af41587517cccb6cfb1367b6ca6fb567184b9a521`
- v2 rubric SHA-256: `01c79af83245e787b98668f034acfc135f52ef95481276cc146c7ee45de1ef28`

The v2 manifest records both archival hashes. The baseline report keeps its original v1 rubric
hash and is not rewritten.
