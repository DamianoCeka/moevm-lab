# Research questions

1. How stable are expert selections across adjacent decode tokens for real workloads?
2. Does routing locality remain useful when prompts change domain or language?
3. Can cross-layer predictions be computed early enough to hide host-to-device transfer?
4. What confidence threshold maximizes latency saved per transferred byte?
5. Should demand, speculative and miss-buffer VRAM be statically partitioned or dynamically resized?
6. When does CPU expert computation beat transferring a missed expert to GPU?
7. Can checkpoint reordering convert random expert reads into large sequential reads?
8. What tile size allows useful compute/transfer overlap without destroying kernel efficiency?
9. How much quality loss is introduced by expert-specific adaptive quantization?
10. Can a scheduler remain beneficial with batching, where more experts activate simultaneously?
11. Which metrics predict end-to-end gains better than raw cache hit-rate?
12. What is the smallest real MoE that reproduces the same memory bottleneck class as a 3T-scale model?
