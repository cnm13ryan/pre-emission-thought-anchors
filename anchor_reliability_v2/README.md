# anchor_reliability_v2

Standalone follow-up to the pre-emission thought-anchor MVP. It asks whether the
counterfactual-importance target can be measured reliably, repairs the measurement
(bounded scores, the v2 answer parser, pre-emission eligibility), and re-evaluates
prediction against stronger baselines.

- The MVP (rest of this repository at `f11823f`) is **read-only input**. This folder never
  imports it; copied code carries its source in the module docstring, and copied data
  carries SHA-256 provenance in `$ARV2_DATA/inputs/MANIFEST.json`.
- Preregistration: [`phase2_plan.md`](phase2_plan.md). Everything on the 20 MVP problems is
  exploratory.

Work in progress; see the status section of the latest report in `results/`.
