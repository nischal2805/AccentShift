# Write tests for a pipeline module

Write pytest tests for the module specified below.

Rules:
- Mock all model calls (never load actual models in tests)
- Use synthetic audio: `np.random.randn(16000).astype(np.float32)` for 1s at 16kHz
- Test happy path + edge cases: silence input, very short audio (<0.5s), very long audio (>30s)
- Test that exceptions are raised correctly on bad input
- Use fixtures for repeated setup

Module to test: $ARGUMENTS
