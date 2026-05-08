# P09H-STAT Feedback Error Audit

## Policy-Level Feedback Error
- VoP: 0.499866
- Periodic: 0.499941
- Random: 0.499935
- Uncertainty-only: 0.499876
- Always-profile: 0.000000

## Absolute Differences: VoP Minus Baseline
- VoP - periodic: -0.000075478
- VoP - random: -0.000069810
- VoP - uncertainty-only: -0.000010648

Negative values mean VoP has lower feedback error. The magnitudes are on the order of 1e-5 to 1e-4, which is tiny relative to the metric scale near 0.5.

## Relative Differences: VoP Minus Baseline
- VoP vs periodic: -0.015097%
- VoP vs random: -0.013964%
- VoP vs uncertainty-only: -0.002130%

## Per-Task Feedback Error Contributions
- task 2: value_of_profile=0.999909, periodic_profile=0.499954, random_profile=0.499954, uncertainty_only=0.000000
- task 6: value_of_profile=0.499916, periodic_profile=0.499916, random_profile=0.499917, uncertainty_only=0.000000
- task 32: value_of_profile=0.000000, periodic_profile=0.499994, random_profile=0.000000, uncertainty_only=0.499994
- task 50: value_of_profile=0.999503, periodic_profile=0.499877, random_profile=0.499877, uncertainty_only=0.999459
- task 88: value_of_profile=0.000000, periodic_profile=0.499964, random_profile=0.999929, uncertainty_only=0.999929

## Metric Sensitivity
The feedback-error metric appears saturated or insensitive in this P10H stream: budgeted policies cluster tightly around 0.4999, while cheap-only is near 0.9999 and always-profile is exactly 0 by construction. This supports a cost-quality or non-inferiority framing, not a superiority claim.

## Can This Support a Superiority Claim?
No. P10H does not support wording such as "VoP significantly beats periodic/random in feedback error." The paper-facing wording should be:

> VoP achieves comparable feedback error to budgeted baselines while reducing method full-profile calls and method wall-clock relative to always-profile.

## Source Files
- `results/search/phase_10h/closed_loop_eval.jsonl`
- `results/search/phase_10h/aggregate_analysis.json`
- `tables/phase_10h/table_closed_loop_policy_summary.csv`
