# P09H-STAT Cost-Quality and Pareto Reframing

## Supported Claims
- Cost reduction versus always-profile: VoP uses 5/10 method full profiles, a 50% reduction.
- Wall-clock reduction versus always-profile: VoP 357.81s versus always-profile 456.17s.
- Comparable feedback quality versus budgeted baselines: VoP, periodic, random, and uncertainty-only are tightly clustered around 0.4999 feedback error.
- Task88-excluded search-quality versus periodic/random: VoP 1.2337x, periodic 0.9957x, random 0.9993x.

## Unsupported Claims
- VoP feedback-error superiority.
- Clear VoP superiority over uncertainty-only.
- ProfileSketch prompt-representation superiority.
- Consistently faster kernels.

## Recommended Figure Framing
Plot full-profile calls or method wall-clock on the x-axis and feedback/search-quality on the y-axis. The visual message should be cost-quality tradeoff, not statistical dominance.
