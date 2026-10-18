# Profiling and Scaling Multi-GPU LLM Training on Google Cloud

Work in progress. Training GPT-2 small (124M) with PyTorch DDP on 1, 2, and 4 GPUs,
measuring scaling, profiling bottlenecks, and documenting what limited each configuration.

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).

The full README (setup, reproduction steps, results, findings, limitations) is written in a later step.
No results are reported until they come from logged runs in `results/`.
