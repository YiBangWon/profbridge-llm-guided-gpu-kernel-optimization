# P09H-STAT Advisor Addendum

## What Changed After Statistical Sanity Check
P10H remains useful, but the paper-facing interpretation should be more conservative than the raw `strong_acceptance_evidence` label. The feedback-error margin between VoP and periodic/random/uncertainty-only is extremely small, and uncertainty-only is nearly tied.

## Safer Interpretation
- Main claim: 50% full-profile reduction plus lower method wall-clock under comparable feedback quality.
- Secondary claim: task88-excluded search-quality is directionally better than periodic/random.
- Caveated claim: VoP is competitive with uncertainty-only, not clearly superior.
- ProfileSketch should be framed as a structured acquisition/control representation, not as a better prompt representation.

## Exact Caveats
- Feedback-error margins are tiny; do not claim statistical superiority.
- Uncertainty-only is nearly tied with VoP.
- ProfileSketch representation ablation is mixed.
- P10H covers five tasks.
- Task88 is influential and task36 remains unresolved.

## One-Minute Explanation
P10H is still helpful, but not because VoP decisively lowers feedback error. The honest story is that VoP cuts method full-profile calls by 50% and lowers method wall-clock versus always-profile while maintaining feedback error comparable to other budgeted policies. The more useful search-quality signal is task88-excluded speedup versus periodic/random, but uncertainty-only is almost tied, so the paper should center ProfileSketch/VoP as acquisition accounting and control rather than as a clearly superior optimizer.

## What Not To Say
- Do not say "VoP significantly beats periodic/random in feedback error."
- Do not say "VoP dominates uncertainty-only."
- Do not say "ProfileSketch is a better prompt representation."
- Do not say "ProfBridge consistently finds faster kernels."

## Decisions Needed From Advisor
1. Is comparable feedback quality plus lower profile cost enough for the main CGO claim?
2. Should uncertainty-only be framed as an ablation that nearly matches VoP?
3. Should P09I add a formal non-inferiority framing or keep it as sensitivity analysis?
4. Should task36 be fixed before submission?
5. Is the current accept estimate of 40-48% honest enough before advisor feedback?
