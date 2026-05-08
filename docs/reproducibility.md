# Reproducibility Notes

Live GPU profiling requires CUDA, PyTorch, NVIDIA Nsight Compute, and KernelBench-style task definitions.

The curated reports and example records can be inspected without new LLM API calls.

For a quick check:

```bash
python -m compileall profbridge scripts
python scripts/smoke_test.py
```
