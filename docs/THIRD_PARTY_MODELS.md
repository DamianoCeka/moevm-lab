# Third-party models and tools

MoEVM Lab does not redistribute model weights.

## OLMoE

The Windows one-command demo may acquire the pinned checkpoint into the user's
local Hugging Face cache. No model weight is committed to or redistributed by
this repository; acquisition is revision-pinned, excludes known legacy weight
formats, and the runtime loads only the three safetensors shards verified by
project-held size and SHA-256 checks. See [One-command OLMoE demo](ONE_COMMAND_DEMO.md).

The v0.2 M1 study uses `allenai/OLMoE-1B-7B-0924`, pinned to commit
`bd1c52f59153f724c1ad11ca1791edc77bab3806`. Its model card and the official
OLMoE repository identify the project as Apache License 2.0:

- [OLMoE model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [OLMoE source repository](https://github.com/allenai/OLMoE)
- [OLMoE paper](https://arxiv.org/abs/2409.02060)

Only derived routing decisions, top-k probabilities and controlled generated
token IDs are stored in this repository. Those short token-ID sequences can be
decoded and may reflect model output or model training material. They are
included only for reproducibility. MoEVM Lab makes no ownership or licensing
claim over decoded underlying text; the project's Apache-2.0 license grants
rights only to material the project is authorized to license. The checkpoint
remains in the local Hugging Face cache and is excluded from Git.

## Capture stack

The optional real-capture environment pins Transformers, Accelerate,
Hugging Face Hub and Safetensors in `pyproject.toml`. PyTorch is installed
separately from the official CUDA wheel index because the correct build depends
on the host GPU and driver.

- [Transformers OLMoE documentation](https://huggingface.co/docs/transformers/model_doc/olmoe)
- [Accelerate big-model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)

These projects retain their respective copyrights and licenses. Their inclusion
here is attribution and workflow documentation, not relicensing.

## Adapted Transformers code

The expert execution loop in `src/moevm/paged_runtime.py` is adapted and
modified from the Apache-2.0-licensed `OlmoeExperts`/`MixtralExperts` forward
implementation in Hugging Face Transformers 5.14.1. The relevant upstream
copyright notice is preserved in that source file and in the project NOTICE.
