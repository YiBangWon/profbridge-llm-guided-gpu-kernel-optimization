# P09H-STAT Bootstrap and Paired Sanity Analysis

## Method
- Unit of resampling: task.
- Feedback metric: mean `feedback_error_contribution` per task and policy.
- Speedup metric: per-task geomean candidate speedup, with task88 excluded for the main speedup sanity check.
- Bootstrap: 10,000 task-resampling replicates with fixed seed.
- Sign test: exact two-sided binomial test on task wins/losses, reported only as a small-n sanity check.

## Results
- `feedback_error_vop_minus_periodic_profile`: mean diff -0.000075478, 95% bootstrap CI [-0.399983237, 0.399832282], wins/losses/ties 2/3/0, sign-test p 1.0000.
- `speedup_without_task88_vop_minus_periodic_profile`: mean diff 0.332795297, 95% bootstrap CI [-0.000707137, 0.986269411], wins/losses/ties 2/2/0, sign-test p 1.0000.
- `feedback_error_vop_minus_random_profile`: mean diff -0.000069810, 95% bootstrap CI [-0.500032220, 0.399897701], wins/losses/ties 2/2/1, sign-test p 1.0000.
- `speedup_without_task88_vop_minus_random_profile`: mean diff 0.329249036, 95% bootstrap CI [-0.000977824, 0.977669764], wins/losses/ties 2/2/0, sign-test p 1.0000.
- `feedback_error_vop_minus_uncertainty_only`: mean diff -0.000010648, 95% bootstrap CI [-0.599972868, 0.599938829], wins/losses/ties 2/3/0, sign-test p 1.0000.
- `speedup_without_task88_vop_minus_uncertainty_only`: mean diff 0.001970133, 95% bootstrap CI [0.000822594, 0.003117671], wins/losses/ties 4/0/0, sign-test p 0.1250.

## Interpretation
The feedback-error differences are tiny and bootstrap intervals overlap zero in this five-task run. This should not be described as statistical superiority. The task88-excluded speedup/search-quality comparisons are more directionally favorable versus periodic/random, but uncertainty-only is close and the sample is too small for strong inference.

## Recommended Wording
- "Comparable feedback error under lower method profile cost."
- "Directionally better task88-excluded search-quality than periodic/random in this run."
- "Uncertainty-only is nearly tied and remains a important comparator."
- "Small-n bootstrap sanity checks do not support formal superiority claims."
