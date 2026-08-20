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

## Qwen1.5-MoE research target

The next full-checkpoint acceptance candidate is
`Qwen/Qwen1.5-MoE-A2.7B`, pinned for reproducibility to commit
`1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`. The official model card describes
14.3 billion total parameters, 2.7 billion activated parameters, 60 routed
experts with top-4 routing and a shared expert. Its eight BF16 safetensors
shards total approximately 28.6 GB.

- [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/tree/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9)
- [Qwen model card](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
- [Qwen checkpoint configuration](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/config.json)
- [Transformers Qwen2MoE documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_moe)
- [Tongyi Qianwen license](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/LICENSE)

The checkpoint weights are licensed separately under the Tongyi Qianwen
License Agreement, not under MoEVM Lab's Apache-2.0 license. Anyone who elects
to acquire or use the checkpoint must review and comply with those terms.
MoEVM Lab does not redistribute the weights, tokenizer assets or other Qwen
checkpoint files, and its license does not grant rights to those third-party
materials.

Current evidence for the pinned full checkpoint is narrow and specific: this path
has a successful deterministic local reference capture (`systems_en`,
`max_new_tokens=2`, `seed=17`, `temperature=0.0`) with full-manifest
integrity verification, and a full-checkpoint sync runtime gate against that
reference that passed exact greedy token parity (2/2 tokens). It is a correctness
smoke only: one short prompt, no throughput claim, and no production-concurrency
coverage.

The deterministic tiny `qwen2_moe` configuration, created locally from synthetic
test weights, also has exact eager-versus-paged logits parity and exercises the
resident shared-expert path.

## Future research candidates

Two public Apache-2.0 checkpoints are retained as possible later research
targets, not as supported models:

- [`ibm-granite/granite-4.0-h-tiny-base`](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base)
  is a 7B-total/1B-active hybrid Mamba2/MoE model with shared experts. Its
  stacked expert tensors and different module path require an audited adapter
  and conversion boundary before MoEVM can evaluate it.
- [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) is a
  20.9B-total/3.6B-active MoE whose expert weights use MXFP4 blocks and scales.
  Supporting it requires an explicit quantized storage and execution design;
  the current BF16 paged path does not establish compatibility.

Listing these checkpoints records roadmap interest only. MoEVM Lab has not run
full-checkpoint correctness, memory or performance studies on either model.

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
