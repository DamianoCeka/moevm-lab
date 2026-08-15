#!/usr/bin/env python3
"""Capture token/layer expert routing from a pinned open OLMoE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moevm.analysis import analyze_routing_trace, write_trace_analysis
from moevm.config import load_config
from moevm.olmoe_assets import (
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_SHARD_SHA256,
)
from moevm.trace import read_trace

DEFAULT_MODEL = PINNED_MODEL_ID
DEFAULT_REVISION = PINNED_REVISION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture real OLMoE router decisions into MoEVM JSONL traces."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--workloads",
        default=str(_REPO_ROOT / "benchmarks" / "workloads" / "olmoe_m1.json"),
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "olmoe_1b_7b_0924.toml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "results" / "real-routing" / "olmoe-0924"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(
            Path(os.environ.get("HF_HOME", _REPO_ROOT / ".cache" / "huggingface"))
            / "hub"
        ),
    )
    parser.add_argument("--offload-dir", default=None)
    parser.add_argument("--workload-id", action="append", default=[])
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-memory", default="9GiB")
    parser.add_argument("--cpu-memory", default="20GiB")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _serialized_device_map(model: Any) -> dict[str, str]:
    """Return a stable device map for dispatched and fully resident models."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping) and device_map:
        return {str(key): str(value) for key, value in device_map.items()}
    device = getattr(model, "device", None)
    return {"": str(device)} if device is not None else {}


def _load_workloads(path: Path, selected_ids: set[str]) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("workloads"), list):
        raise ValueError("workload file must contain a workloads array")
    workloads: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["workloads"]):
        if not isinstance(item, dict):
            raise ValueError(f"workload {index} must be an object")
        required = ("id", "category", "language", "prompt")
        if any(not isinstance(item.get(field), str) for field in required):
            raise ValueError(f"workload {index} has missing or invalid string fields")
        workload = {field: item[field] for field in required}
        workload_id = workload["id"]
        if re.fullmatch(r"[A-Za-z0-9_-]+", workload_id) is None:
            raise ValueError(
                f"workload id must use only letters, digits, '_' or '-': {workload_id}"
            )
        if workload_id in seen:
            raise ValueError(f"duplicate workload id: {workload_id}")
        seen.add(workload_id)
        if not selected_ids or workload_id in selected_ids:
            workloads.append(workload)
    missing = selected_ids - seen
    if missing:
        raise ValueError(f"unknown workload ids: {', '.join(sorted(missing))}")
    if not workloads:
        raise ValueError("no workloads selected")
    return workloads


def _sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _router_rows(
    torch: Any,
    router_logits: tuple[Any, ...] | list[Any] | None,
    *,
    token_start: int,
    token_ids: list[int],
    phase: str,
    top_k: int,
    experts_per_layer: int,
) -> list[dict[str, object]]:
    if router_logits is None:
        raise RuntimeError("model did not return router_logits")
    rows: list[dict[str, object]] = []
    expected_tokens = len(token_ids)
    for layer, logits in enumerate(router_logits):
        if logits.ndim != 2 or logits.shape[0] != expected_tokens:
            raise RuntimeError(
                f"unexpected router shape at layer {layer}: {tuple(logits.shape)}"
            )
        if logits.shape[1] != experts_per_layer:
            raise RuntimeError(
                f"unexpected expert count at layer {layer}: {logits.shape[1]}"
            )
        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        scores, experts = torch.topk(probabilities, top_k, dim=-1)
        for offset, (expert_row, score_row) in enumerate(
            zip(experts.cpu().tolist(), scores.cpu().tolist(), strict=True)
        ):
            rows.append(
                {
                    "token": token_start + offset,
                    "layer": layer,
                    "experts": expert_row,
                    "scores": score_row,
                    "token_id": token_ids[offset],
                    "phase": phase,
                }
            )
    return sorted(rows, key=lambda row: (int(row["token"]), int(row["layer"])))


def _write_trace_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def _select_token(torch: Any, logits: Any, temperature: float) -> Any:
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _package_versions() -> dict[str, str]:
    packages = (
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def _download_snapshot(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    local_files_only: bool,
    hf_api: Any,
    hf_hub_download: Any,
    snapshot_download: Any,
) -> Path:
    ignored_suffixes = (".bin", ".h5", ".msgpack", ".ot")
    try:
        return Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                ignore_patterns=tuple(f"*{suffix}" for suffix in ignored_suffixes),
            )
        )
    except OSError as exc:
        if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
            raise
        # Some Windows hosts disallow symlinks. huggingface_hub may have already
        # copied large blobs into the snapshot before one concurrent symlink
        # attempt fails. Complete only missing files directly in that directory.
        snapshot_path = (
            cache_dir
            / f"models--{model_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snapshot_path.mkdir(parents=True, exist_ok=True)
        api = hf_api()
        for filename in api.list_repo_files(model_id, revision=revision):
            if filename.endswith(ignored_suffixes):
                continue
            target = snapshot_path / filename
            if target.is_file():
                continue
            print(f"Windows no-symlink fallback: fetching {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                local_dir=snapshot_path,
                local_files_only=local_files_only,
            )
        return snapshot_path


def _verify_pinned_shards(snapshot_path: Path) -> dict[str, str]:
    verified: dict[str, str] = {}
    for filename, expected in PINNED_SHARD_SHA256.items():
        path = snapshot_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {filename}: {actual} != {expected}"
            )
        verified[filename] = actual
        print(f"Verified {filename}: {actual}")
    return verified


def _capture_workload(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    workload: dict[str, str],
    args: argparse.Namespace,
    workload_path: Path,
    output_dir: Path,
    config: Any,
) -> dict[str, object]:
    encoded = tokenizer(
        workload["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(input_device)
    prompt_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    if not prompt_ids:
        raise RuntimeError(f"tokenizer produced no tokens for {workload['id']}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync_cuda(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            output_router_logits=True,
            logits_to_keep=1,
        )
    _sync_cuda(torch)
    prefill_seconds = time.perf_counter() - started
    rows = _router_rows(
        torch,
        output.router_logits,
        token_start=0,
        token_ids=prompt_ids,
        phase="prefill",
        top_k=config.model.top_k,
        experts_per_layer=config.model.experts_per_layer,
    )

    generated_ids: list[int] = []
    decode_forward_seconds: list[float] = []
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for decode_index in range(args.max_new_tokens):
            next_token = _select_token(torch, output.logits[:, -1, :], args.temperature)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            _sync_cuda(torch)
            forward_started = time.perf_counter()
            output = model(
                input_ids=next_token.to(input_device),
                past_key_values=output.past_key_values,
                use_cache=True,
                output_router_logits=True,
                logits_to_keep=1,
            )
            _sync_cuda(torch)
            decode_forward_seconds.append(time.perf_counter() - forward_started)
            rows.extend(
                _router_rows(
                    torch,
                    output.router_logits,
                    token_start=len(prompt_ids) + decode_index,
                    token_ids=[token_id],
                    phase="decode",
                    top_k=config.model.top_k,
                    experts_per_layer=config.model.experts_per_layer,
                )
            )
    _sync_cuda(torch)
    decode_seconds = time.perf_counter() - decode_started
    generation_decode_seconds = sum(decode_forward_seconds[:-1])
    generation_wall_seconds = prefill_seconds + generation_decode_seconds

    trace_path = output_dir / f"{workload['id']}.trace.jsonl"
    _write_trace_rows(trace_path, rows)
    trace_steps = read_trace(trace_path)
    analysis = analyze_routing_trace(
        trace_steps,
        experts_per_layer=config.model.experts_per_layer,
        predictor_config=config.predictor,
    )
    analysis_dir = output_dir / f"{workload['id']}.analysis"
    analysis_json, analysis_markdown = write_trace_analysis(
        analysis_dir,
        analysis,
        trace_path=trace_path,
    )

    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "evidence_label": "routing capture",
        "workload": {
            "id": workload["id"],
            "category": workload["category"],
            "language": workload["language"],
            "prompt_sha256": hashlib.sha256(
                workload["prompt"].encode("utf-8")
            ).hexdigest(),
            "workload_file": str(workload_path),
            "workload_file_sha256": _sha256(workload_path),
        },
        "model": {
            "id": args.model,
            "requested_revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "license": "Apache-2.0",
            "layers": config.model.layers,
            "experts_per_layer": config.model.experts_per_layer,
            "top_k": config.model.top_k,
            "checkpoint_shards_sha256": getattr(args, "checkpoint_shards_sha256", None),
        },
        "generation": {
            "seed": args.seed,
            "temperature": args.temperature,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
        },
        "timing_observation": {
            "warning": "Model execution with Accelerate CPU offload; not an MoEVM runtime benchmark.",
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_forward_seconds": decode_forward_seconds,
            "generation_decode_seconds": generation_decode_seconds,
            "routing_only_final_forward_seconds": (
                decode_forward_seconds[-1] if decode_forward_seconds else 0.0
            ),
            "generation_wall_seconds": generation_wall_seconds,
            "generation_tokens_per_second_including_prefill": (
                len(generated_ids) / generation_wall_seconds
                if generation_wall_seconds
                else 0.0
            ),
            "decode_tokens_per_second": (
                (len(generated_ids) - 1) / generation_decode_seconds
                if len(generated_ids) > 1 and generation_decode_seconds
                else 0.0
            ),
        },
        "trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            "steps": len(rows),
            "analysis_json": str(analysis_json),
            "analysis_markdown": str(analysis_markdown),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_vram_bytes": peak_vram_bytes,
            "device_map": _serialized_device_map(model),
        },
    }
    metadata_path = output_dir / f"{workload['id']}.metadata.json"
    _write_json(metadata_path, metadata)
    print(
        f"Captured {workload['id']}: {analysis.tokens} tokens, "
        f"{len(rows)} layer steps, trace {metadata['trace']['sha256']}"
    )
    return {
        "workload_id": workload["id"],
        "trace": str(trace_path),
        "trace_sha256": metadata["trace"]["sha256"],
        "metadata": str(metadata_path),
        "analysis": str(analysis_json),
    }


def main() -> int:
    args = _parse_args()
    if args.max_input_tokens <= 0 or args.max_new_tokens < 0:
        raise ValueError("token limits must be positive (new tokens may be zero)")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    workload_path = Path(args.workloads).resolve()
    workloads = _load_workloads(workload_path, set(args.workload_id))
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "real capture dependencies are missing; install a CUDA PyTorch wheel "
            "and then `python -m pip install -e .[real-traces]`"
        ) from exc

    print(f"Model: {args.model}@{args.revision}")
    print(f"Cache: {cache_dir}")
    snapshot_path = _download_snapshot(
        model_id=args.model,
        revision=args.revision,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        hf_api=HfApi,
        hf_hub_download=hf_hub_download,
        snapshot_download=snapshot_download,
    )
    print(f"Pinned snapshot: {snapshot_path}")
    if args.model == DEFAULT_MODEL and args.revision == DEFAULT_REVISION:
        args.checkpoint_shards_sha256 = _verify_pinned_shards(snapshot_path)
    else:
        args.checkpoint_shards_sha256 = None
        print("No built-in checkpoint hashes are available for this custom revision.")
    if args.download_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this M1 capture")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = load_config(args.config)
    offload_dir = Path(
        args.offload_dir or (Path(args.output_dir) / "offload")
    ).resolve()
    offload_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
    )
    print("Loading checkpoint with GPU/CPU dispatch...")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model_load_seconds = time.perf_counter() - load_started
    print(f"Loaded checkpoint in {model_load_seconds:.3f} seconds")
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision not in (None, args.revision):
        raise RuntimeError(
            f"resolved model revision {resolved_revision} != requested {args.revision}"
        )
    if model.config.num_hidden_layers != config.model.layers:
        raise RuntimeError("model layer count does not match the replay config")
    if model.config.num_experts != config.model.experts_per_layer:
        raise RuntimeError("model expert count does not match the replay config")
    if model.config.num_experts_per_tok != config.model.top_k:
        raise RuntimeError("model top-k does not match the replay config")

    output_dir = Path(args.output_dir).resolve() / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _capture_workload(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            workload=workload,
            args=args,
            workload_path=workload_path,
            output_dir=output_dir,
            config=config,
        )
        for workload in workloads
    ]
    manifest = {
        "schema_version": 1,
        "evidence_label": "routing capture",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "temperature": args.temperature,
        "model_load_seconds": model_load_seconds,
        "dispatch": {
            "gpu_memory": args.gpu_memory,
            "cpu_memory": args.cpu_memory,
            "dtype": "bfloat16",
            "device_map": "auto",
        },
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
