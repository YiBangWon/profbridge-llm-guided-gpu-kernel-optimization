# Causal Ablation: Does Profiler-Derived Bottleneck Guidance Cause Better Search?

**Status:** curated finding · honest / falsification-style result
**Data:** phase 10e + phase 10g controlled causal-ablation evaluations (real `gpt-5.5`
generations), KernelBench-style Level-1 tasks, 85 candidates before gating.

## Question

If we inject profiler-derived bottleneck information into the LLM prompt, do we get
faster candidates *because of the profiler signal* — or because the prompt simply
*frames the task as an optimization problem*?

## Method

All arms share the same task set and the same parent candidate. The only thing that
changes between arms is the bottleneck information the prompt receives:

- `control_static` — no bottleneck information (baseline, ratio defined = 1.0).
- `guided_predicted_bottleneck` — predicted bottleneck label + numeric signal.
- `prompt_only_framing` — optimization framing only, no profiler-derived content.
- `oracle_measured_bottleneck` — ground-truth measured bottleneck (intended ceiling).
- `corrupted_wrong` / `corrupted_random` / `corrupted_noisy` — deliberately degraded
  bottleneck signals (intended floor).
- `guided_label_only`, `action_shaping`, `action_shaping_wrong` — secondary arms.

**Honesty gate (applied before any metric):** a candidate counts only if it passes
correctness, robust-correctness, the loophole scanner, the safety scanner, and Nsight
profiling success. Gated counts are reported per arm so the filtering is auditable.

**Statistic:** per-arm geometric-mean speedup (over eager, and normalized to the
static control for the same task), plus a one-sided paired bootstrap (20k resamples,
paired by task) on the per-task geomean ratio between two arms.

## Result

| Contrast (paired by task, gated) | Geomean ratio | 95% CI | p (A not better) |
|---|---:|---:|---:|
| predicted-bottleneck vs static control | 1.185 | [0.99, 1.66] | 0.118 |
| prompt-only framing vs static control | 1.184 | [1.00, 1.65] | 0.063 |
| predicted-bottleneck vs prompt-only framing | 1.000 | [0.99, 1.01] | 0.480 |
| oracle measured-bottleneck vs static control | 1.095 | [1.00, 1.29] | 0.000 |

Per-arm geomean speedup vs static control (honesty-gated):

- control_static ≈ 1.000
- corrupted_wrong ≈ 1.002, corrupted_random ≈ 1.008
- corrupted_noisy ≈ 1.094, oracle_measured_bottleneck ≈ 1.095
- prompt_only_framing ≈ 1.184, guided_predicted_bottleneck ≈ 1.185
- guided_label_only ≈ 1.237 (n_gated = 4, low power)

## Interpretation

1. Predicted-bottleneck guidance is the top "real-signal" arm (+18% geomean over the
   static control). Taken alone this looks like a positive result.
2. But **prompt-only framing matches it almost exactly** (ratio 1.000, p = 0.48). The
   structured profiler-derived numeric content adds essentially nothing over a prompt
   that merely frames the task as optimization.
3. A clean causal story would require `oracle > predicted > corrupted`. Instead
   `corrupted_noisy ≈ oracle`, and framing dominates — so the gain is **not** caused by
   bottleneck-signal correctness.
4. With only 5 paired tasks, the predicted-vs-static contrast is **not** statistically
   significant (CI crosses 1.0).

**Honest conclusion:** in this prototype, generic optimization *framing* — not
profiler-derived numeric bottleneck content — explains essentially all of the measured
search-quality gain. The intuitive "profiler signal helps the LLM" hypothesis is not
supported by this controlled ablation.

## Why this is reported, not buried

This is the central scientific value of the project: a controlled design with
floor/ceiling arms, an automated honesty gate applied before metrics, and paired
bootstrap inference — used to *falsify* an attractive hypothesis rather than to
decorate a result. The reusable infrastructure (Nsight wrapper, ProfileSketch,
honesty gate, paired-bootstrap analysis) stands independently of the negative finding.

## Future direction

Design prompts that explicitly separate "the prompt mentions optimization" from "the
prompt carries profiler-derived numeric signal," with enough tasks for statistical
power, before claiming profiler-guided search beats generic search.

Reproduce from curated artifacts: `python scripts/analyze_bottleneck_guided_search_quality.py`
(reads `tables/curated/bottleneck_guided_retro_summary.json`, re-renders
`figures/curated/fig_bottleneck_guided_search_quality.png`).
