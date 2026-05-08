# ProfileSketch Schema

A ProfileSketch is a compact typed record with:

- task and candidate identity,
- cheap features,
- predicted metric estimates,
- uncertainty,
- measured metrics when available,
- bottleneck labels,
- acquisition scores,
- guidance metadata,
- provenance.

Missing metrics remain missing and should not be zero-filled.
