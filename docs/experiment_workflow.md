# Experiment Workflow

1. Run a dependency-light smoke test.
2. Evaluate a baseline or candidate on a KernelBench-style task.
3. Optionally collect selected NVIDIA Nsight Compute metrics.
4. Build profile-pair records and ProfileSketch records.
5. Evaluate profile acquisition policies under a fixed budget.
6. Run statistical sanity checks before making claims.
