# Method Overview

ProfBridge studies profiler feedback as a budgeted resource in LLM-guided GPU kernel optimization.

The project separates three questions:

1. How do we evaluate generated GPU candidates safely?
2. How do we represent profiler feedback compactly?
3. When should a search loop pay for full profiler feedback?

The core abstraction is ProfileSketch. The core policy is Value-of-Profile.
