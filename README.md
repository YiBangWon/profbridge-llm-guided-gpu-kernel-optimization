# ProfBridge: LLM-Guided GPU Kernel Optimization

> A research prototype for evaluating and analyzing LLM-generated GPU optimization candidates with NVIDIA Nsight Compute profiler feedback.

ProfBridge is not intended to compete with production GPU kernel generators. The goal is to study profiler-aware feedback, evaluation infrastructure, and claim calibration for LLM-guided GPU optimization research.

## Abstract

LLM-guided compiler optimization systems can generate many candidate GPU implementations, but evaluating every candidate with a high-fidelity profiler is expensive. This project explores whether profiler feedback can be represented, predicted, and selectively acquired under a limited profiling budget.

**ProfBridge** builds an end-to-end research prototype around KernelBench-style GPU operators, PyTorch, NVIDIA Nsight Compute, and LLM-generated candidate code. It introduces a structured feedback representation called `ProfileSketch` and evaluates `Value-of-Profile`, a policy that decides whether a candidate should receive full profiler feedback or use predicted feedback instead.

This repository is not a finished paper. The final conclusion is deliberately conservative: reducing profiling calls alone is not enough for a strong compiler paper. The most useful outcome is an experimental infrastructure for GPU profiling, candidate evaluation, feedback representation, and claim calibration.

## Motivation

Recent LLM-guided compiler systems ask:

> Which optimization candidate should we generate next?

This project studied a related systems question:

> If an LLM generates many candidates, do we need to fully profile every candidate?

Full GPU profiling with NVIDIA Nsight Compute provides hardware-level signals such as memory traffic, instruction count, and warp activity, but it is much more expensive than simple latency measurement or static feature extraction. ProfBridge treats this profiler feedback as a scarce resource.

## Background

This project sits near three research areas:

- **Tensor compiler autotuning** such as TVM, AutoTVM, Ansor, and MetaSchedule.
- **LLM-guided compiler search** such as Reasoning Compiler and Autocomp.
- **Profiler-guided GPU kernel optimization** systems that use measured hardware feedback.

ProfBridge does not aim to be a better code generator than these systems. Instead, it focuses on a narrower question: how expensive profiler feedback can be represented, acquired, and analyzed inside such search loops.

## Key Idea

ProfBridge uses a compact structured representation called `ProfileSketch`.

A `ProfileSketch` contains:

- predicted profiler metrics,
- uncertainty estimates,
- bottleneck labels,
- signal provenance,
- acquisition decision metadata.

A `Value-of-Profile` policy then decides whether full profiling is worth paying for.

```text
candidate code
    -> cheap timing / static features
    -> ProfileSketch
    -> Value-of-Profile decision
    -> full profile or predicted feedback
    -> next optimization step
```

## What I Built

This repository contains:

- KernelBench-style GPU candidate evaluation utilities,
- PyTorch eager and candidate timing/profiling harnesses,
- NVIDIA Nsight Compute wrapper and selected metric parser,
- profile-pair schema and validation utilities,
- ProfileSketch representation,
- Value-of-Profile policy evaluator,
- closed-loop policy analysis scripts,
- statistical sanity checks,
- robust correctness and candidate-loophole scanner,
- curated reports, tables, and figures.

## Experiments

The prototype compares several profiling policies:

| Policy | Description |
|---|---|
| always-profile | Full profiling for every candidate |
| cheap-only | No full profiling |
| periodic-profile | Profile at fixed intervals |
| random-profile | Profile randomly under budget |
| uncertainty-only | Profile high-uncertainty candidates |
| Value-of-Profile | Profile candidates with high estimated profiling value |

Metrics include full profiler calls, method wall-clock, feedback error, search-quality/speedup, correctness pass rate, and Nsight Compute success rate.

## Key Results

The strongest result is not that Value-of-Profile significantly improves feedback error. In fact, the feedback-error margins are very small.

The safer interpretation is:

> ProfBridge can reduce full profiling cost while maintaining comparable feedback quality in a bounded prototype.

Example closed-loop result:

| Metric | always-profile | Value-of-Profile |
|---|---:|---:|
| full profiles | 10 | 5 |
| method wall-clock | 456.17s | 357.81s |
| feedback error | 0.0 | 0.499866 |
| valid candidates | - | 105/105 |

Important caveat: budgeted baselines had very similar feedback error values, so this project does **not** claim statistically significant feedback-error superiority.

## What Did Not Work

Several hypotheses became weaker after experiments:

1. **Profiling reduction alone is not a strong compiler-paper contribution.** Many autotuning systems already reduce measurement cost.
2. **Value-of-Profile did not clearly dominate uncertainty-only.** Uncertainty-only was a strong baseline.
3. **ProfileSketch was not uniformly better as a prompt representation.** Its stronger role is as a structured representation for acquisition, accounting, and provenance.
4. **Speedup causality was mixed.** Some bottleneck-guided candidates improved performance, but noisy or prompt-only variants were often close.
5. **The research direction needed reframing.** The stronger future direction is not "profile less," but "use profiler-derived bottleneck signals to help LLM search find candidates that generic search misses."

## Lessons Learned

This project produced both working infrastructure and a negative/partial research result.

Main lessons:

- A compiler paper needs a performance-seeking story, not only overhead reduction.
- LLM-generated candidate search must be evaluated against strong baselines.
- Feedback-error differences must be statistically meaningful before being claimed.
- Profiling cost should be treated as a constraint or secondary metric unless it directly improves search quality.
- Honest claim calibration is more important than optimistic labeling.

## Repository Structure

```text
profbridge/
|-- profbridge/       # core Python package
|-- scripts/          # experiment and analysis scripts
|-- configs/          # experiment configs
|-- docs/             # method and workflow documentation
|-- reports/curated/  # curated experiment reports
|-- tables/curated/   # summarized result tables
|-- figures/curated/  # curated figures
`-- examples/         # small example records
```

## How to Run

### 1. Environment check

```bash
python -m compileall profbridge scripts
python scripts/smoke_test.py
```

### 2. Build ProfileSketch records from examples

```bash
python scripts/build_profile_sketches.py \
  --input examples/minimal_profile_pair.jsonl \
  --out results/profile_sketches.jsonl
```

### 3. Run Value-of-Profile replay

```bash
python scripts/run_value_of_profile_eval.py \
  --profile-sketches examples/sample_profilesketches.jsonl \
  --profile-budget-fraction 0.5 \
  --policy always_profile,cheap_only,periodic_profile,uncertainty_only,value_of_profile
```

### 4. Inspect statistical sanity analysis

```bash
python scripts/analyze_p10h_statistical_sanity.py
```

The public release ships curated statistical summaries rather than the full private phase logs. The script therefore reads `tables/curated/` by default and prints the conservative paper-facing interpretation.

## Reproducibility Notes

This project depends on:

- CUDA-enabled PyTorch for live GPU evaluation,
- NVIDIA Nsight Compute for full profiling,
- KernelBench or KernelBench-style task files,
- optional LLM API access for generating new candidates.

Most curated analysis can be inspected without new LLM calls using the included reports, tables, and example records.

## Limitations

This is a research prototype, not a final paper.

Known limitations:

- small task count in some live experiments,
- one task had strong influence on aggregate speedup results,
- one task remained unresolved due to constructor/signature mismatch,
- feedback-error margins were tiny,
- uncertainty-only was a very strong baseline,
- ProfileSketch representation ablation was mixed,
- no claim of beating Autocomp, Reasoning Compiler, or production GPU kernel generators.

## Current Status

The original paper framing, "reduce profiler calls with ProfileSketch and Value-of-Profile," is not strong enough by itself for a top compiler paper.

The useful outcome is a research infrastructure and a clearer future direction:

> Use profiler-derived bottleneck signals to guide LLM search toward performance-improving candidates that generic prompts, static rules, or simple acquisition policies miss.

This repository is kept as a research prototype and portfolio artifact demonstrating ML systems/compiler experimentation, GPU profiling, LLM-guided candidate evaluation, and honest claim calibration.
