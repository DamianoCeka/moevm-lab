"""Read-only structural inspection for local safetensors MoE checkpoints.

The inspector intentionally uses only the Python standard library.  It parses
``config.json``, a Hugging Face safetensors index (or one unsharded
``model.safetensors`` file), and safetensors headers.  It never imports model
code, materializes tensor payloads, contacts a network service, or writes to
the snapshot.

The first supported structural contract is the per-expert gated MLP layout
used by Qwen2MoE and by MoEVM's normalized expert store.  Legacy Mixtral
``w1``/``w2``/``w3`` names are recognized as a second, explicit naming scheme
so callers can distinguish a normalizable checkpoint from an unknown layout.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_CONFIG_MAX_BYTES = 4 * 1024 * 1024
_INDEX_MAX_BYTES = 64 * 1024 * 1024
_HEADER_MAX_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_NAME_BYTES = 4096
_MAX_SHARDS = 512
_MAX_TENSORS = 524_288
_MAX_AGGREGATE_HEADER_BYTES = 128 * 1024 * 1024

# These are parser safety limits, not model-format limits.  They are deliberately
# far above the supported checkpoints while bounding tuples and validation loops
# derived from an untrusted config.json.
_MAX_HIDDEN_LAYERS = 4096
_MAX_EXPERTS_PER_LAYER = 16_384
_MAX_EXPECTED_EXPERTS = 131_072

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

_CONFIG_DTYPES = {
    "bool": "BOOL",
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "float16": "F16",
    "half": "F16",
    "bfloat16": "BF16",
    "int32": "I32",
    "float32": "F32",
    "float": "F32",
    "int64": "I64",
    "float64": "F64",
    "double": "F64",
}

_ROOT = r"(?P<root>(?:[^.]+\.)*model\.layers)"
_CANONICAL_EXPERT = re.compile(
    rf"^{_ROOT}\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_LEGACY_MIXTRAL_EXPERT = re.compile(
    rf"^{_ROOT}\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.weight$"
)
_SHARED_EXPERT = re.compile(
    rf"^{_ROOT}\.(?P<layer>\d+)\.mlp\.shared_expert\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_SHARED_EXPERT_GATE = re.compile(
    rf"^{_ROOT}\.(?P<layer>\d+)\.mlp\.shared_expert_gate\.weight$"
)

_CANONICAL_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEGACY_PROJECTION_MAP = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


@dataclass(frozen=True, slots=True)
class ModelInspection:
    """Validated model fields needed to interpret routed expert weights."""

    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int
    dense_intermediate_size: int | None
    expert_intermediate_size: int
    shared_expert_intermediate_size: int | None
    hidden_layers: int
    expected_expert_layers: tuple[int, ...]
    experts_per_layer: int
    experts_per_token: int
    hidden_act: str
    declared_dtype: str | None


@dataclass(frozen=True, slots=True)
class ExpertPlacement:
    """Physical placement of one logical routed expert's three projections."""

    layer: int
    expert: int
    shards: tuple[str, ...]
    colocated: bool
    contiguous: bool
    projection_order: tuple[str, ...]
    logical_bytes: int
    span_bytes: int | None
    gap_bytes: int | None


@dataclass(frozen=True, slots=True)
class ExpertInspection:
    """Coverage, shape, dtype, byte size, and placement of routed experts."""

    naming: str
    layers: tuple[int, ...]
    projections: tuple[str, ...]
    dtype: str
    hidden_size: int
    intermediate_size: int
    experts_per_layer: int
    total_experts: int
    tensor_count: int
    bytes_per_expert: int
    logical_bytes: int
    colocated_experts: int
    contiguous_experts: int
    split_experts: int
    placements: tuple[ExpertPlacement, ...]


@dataclass(frozen=True, slots=True)
class SharedExpertInspection:
    """Resident shared-expert projections and their router gate, when present."""

    present: bool
    layers: tuple[int, ...]
    dtype: str | None
    hidden_size: int
    intermediate_size: int | None
    projection_tensor_count: int
    gate_tensor_count: int
    bytes_per_layer: int | None
    logical_bytes: int
    gate_logical_bytes: int


@dataclass(frozen=True, slots=True)
class ShardInspection:
    """Header-only inventory for one referenced safetensors shard."""

    filename: str
    file_size_bytes: int
    header_size_bytes: int
    tensor_count: int
    logical_bytes: int
    expert_tensor_count: int
    shared_expert_tensor_count: int


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """JSON-ready result of one local, read-only checkpoint inspection."""

    schema: str
    schema_version: int
    snapshot_path: str
    config_path: str
    index_path: str | None
    index_kind: str
    read_only: bool
    network_used: bool
    model_code_executed: bool
    model: ModelInspection
    experts: ExpertInspection
    shared_expert: SharedExpertInspection
    shards: tuple[ShardInspection, ...]
    tensor_count: int
    logical_tensor_bytes: int
    declared_total_size_bytes: int | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a structure accepted directly by :func:`json.dumps`."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _TensorMetadata:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    offset_start: int
    offset_end: int
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _ShardHeader:
    filename: str
    file_size_bytes: int
    header_size_bytes: int
    tensors: dict[str, _TensorMetadata]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(
    payload: dict[str, Any], key: str, *, maximum: int | None = None
) -> int:
    value = payload.get(key)
    if not _is_int(value) or value <= 0:
        raise ValueError(f"config {key} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"config {key} exceeds inspector safety limit of {maximum}")
    return value


def _optional_positive_int(
    payload: dict[str, Any], key: str, *, maximum: int | None = None
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not _is_int(value) or value <= 0:
        raise ValueError(f"config {key} must be a positive integer when present")
    if maximum is not None and value > maximum:
        raise ValueError(f"config {key} exceeds inspector safety limit of {maximum}")
    return value


def _read_json_object(path: Path, *, name: str, max_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"{name} size is invalid: {size} bytes")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be a JSON object")
    return payload


def _normalized_input_path(raw_path: str | os.PathLike[str], *, name: str) -> Path:
    if isinstance(raw_path, str) and not raw_path.strip():
        raise ValueError(f"{name} cannot be empty")
    try:
        return Path(raw_path).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a filesystem path") from exc


def _checkpoint_allowed_roots(snapshot: Path) -> tuple[Path, ...]:
    """Return roots that a checkpoint artifact may resolve beneath.

    A normal snapshot is confined to itself.  A canonical Hugging Face cache
    revision (``<repo>/snapshots/<revision>``) may additionally reference the
    real files in that same repository's non-symlink ``blobs`` directory.
    """

    snapshot_root = snapshot.resolve()
    roots = [snapshot_root]
    if snapshot_root.parent.name != "snapshots":
        return tuple(roots)

    repository_root = snapshot_root.parent.parent
    blobs = repository_root / "blobs"
    is_junction = getattr(blobs, "is_junction", lambda: False)
    if not blobs.is_dir() or blobs.is_symlink() or is_junction():
        return tuple(roots)
    resolved_blobs = blobs.resolve()
    if resolved_blobs.parent == repository_root:
        roots.append(resolved_blobs)
    return tuple(roots)


def _confined_checkpoint_file(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
    name: str,
) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"{name} escapes allowed checkpoint roots: {path}")
    return resolved


def _safe_shard_path(snapshot: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("safetensors index contains an invalid shard filename")
    posix = PurePosixPath(filename)
    windows = PureWindowsPath(filename)
    if (
        posix.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or "\\" in filename
        or ":" in filename
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise ValueError(f"safetensors shard escapes snapshot: {filename!r}")
    lexical = snapshot.joinpath(*posix.parts)
    allowed_roots = _checkpoint_allowed_roots(snapshot)
    if not lexical.parent.resolve().is_relative_to(allowed_roots[0]):
        raise ValueError(f"safetensors shard escapes snapshot: {filename!r}")
    return _confined_checkpoint_file(
        lexical,
        allowed_roots=allowed_roots,
        name="safetensors shard",
    )


def _resolve_inputs(
    snapshot_or_index: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None,
) -> tuple[Path, Path, Path | None, Path | None]:
    supplied = _normalized_input_path(snapshot_or_index, name="snapshot_or_index")
    if supplied.is_dir():
        snapshot = supplied
        index = snapshot / "model.safetensors.index.json"
        single = snapshot / "model.safetensors"
        index_path = index if index.is_file() else None
        single_path = single if index_path is None and single.is_file() else None
    elif supplied.is_file() and supplied.suffix == ".safetensors":
        snapshot = supplied.parent
        index_path = None
        single_path = supplied
    elif supplied.is_file() and supplied.name.endswith(".safetensors.index.json"):
        snapshot = supplied.parent
        index_path = supplied
        single_path = None
    else:
        raise ValueError(
            "snapshot_or_index must be a snapshot directory, safetensors index, "
            "or unsharded .safetensors file"
        )
    if index_path is None and single_path is None:
        raise FileNotFoundError(
            f"no model.safetensors.index.json or model.safetensors in {snapshot}"
        )
    allowed_roots = _checkpoint_allowed_roots(snapshot)
    if index_path is not None:
        index_path = _confined_checkpoint_file(
            index_path,
            allowed_roots=allowed_roots,
            name="safetensors index",
        )
    if single_path is not None:
        single_path = _confined_checkpoint_file(
            single_path,
            allowed_roots=allowed_roots,
            name="safetensors shard",
        )
    if config_path is not None:
        # An explicit config is a separate user-authorized input and need not
        # live inside the checkpoint repository.
        config = _normalized_input_path(config_path, name="config_path")
    else:
        config = _confined_checkpoint_file(
            snapshot / "config.json",
            allowed_roots=allowed_roots,
            name="checkpoint config",
        )
    return snapshot, config, index_path, single_path


def _model_inspection(config: dict[str, Any]) -> ModelInspection:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("config model_type must be a non-empty string")
    raw_architectures = config.get("architectures", [])
    if not isinstance(raw_architectures, list) or not all(
        isinstance(item, str) and item for item in raw_architectures
    ):
        raise ValueError("config architectures must be a list of strings")
    hidden_layers = _positive_int(
        config, "num_hidden_layers", maximum=_MAX_HIDDEN_LAYERS
    )
    hidden_size = _positive_int(config, "hidden_size")
    experts = _optional_positive_int(
        config, "num_experts", maximum=_MAX_EXPERTS_PER_LAYER
    )
    if experts is None:
        experts = _positive_int(
            config, "num_local_experts", maximum=_MAX_EXPERTS_PER_LAYER
        )
    if model_type == "qwen2_moe":
        expert_intermediate = _positive_int(config, "moe_intermediate_size")
    else:
        expert_intermediate = _positive_int(config, "intermediate_size")
    top_k = _positive_int(config, "num_experts_per_tok")
    if top_k > experts:
        raise ValueError("config num_experts_per_tok exceeds the expert count")
    hidden_act = config.get("hidden_act")
    if not isinstance(hidden_act, str) or not hidden_act:
        raise ValueError("config hidden_act must be a non-empty string")

    sparse_step = config.get("decoder_sparse_step", 1)
    if not _is_int(sparse_step) or sparse_step <= 0:
        raise ValueError("config decoder_sparse_step must be a positive integer")
    sparse_layer_count = hidden_layers // sparse_step
    if sparse_layer_count == 0:
        raise ValueError("config selects no sparse expert layers")
    expected_expert_count = sparse_layer_count * experts
    if expected_expert_count > _MAX_EXPECTED_EXPERTS:
        raise ValueError(
            "config sparse layer/expert inventory exceeds inspector safety limit "
            f"of {_MAX_EXPECTED_EXPERTS} routed experts"
        )
    # Construct this only after all attacker-controlled count limits pass.  The
    # arithmetic progression is equivalent to filtering every hidden layer but
    # does not scan dense-only layers.
    expected_layers = tuple(range(sparse_step - 1, hidden_layers, sparse_step))

    declared_dtype_raw = config.get("torch_dtype", config.get("dtype"))
    declared_dtype: str | None = None
    if declared_dtype_raw is not None:
        if not isinstance(declared_dtype_raw, str):
            raise ValueError("config dtype must be a string when present")
        declared_dtype = _CONFIG_DTYPES.get(declared_dtype_raw.lower())
        if declared_dtype is None:
            raise ValueError(f"unsupported config dtype: {declared_dtype_raw}")

    dense_intermediate = _optional_positive_int(config, "intermediate_size")
    shared_intermediate = _optional_positive_int(
        config, "shared_expert_intermediate_size"
    )
    return ModelInspection(
        model_type=model_type,
        architectures=tuple(raw_architectures),
        hidden_size=hidden_size,
        dense_intermediate_size=dense_intermediate,
        expert_intermediate_size=expert_intermediate,
        shared_expert_intermediate_size=shared_intermediate,
        hidden_layers=hidden_layers,
        expected_expert_layers=expected_layers,
        experts_per_layer=experts,
        experts_per_token=top_k,
        hidden_act=hidden_act,
        declared_dtype=declared_dtype,
    )


def _weight_map(
    index: dict[str, Any],
) -> tuple[dict[str, str], int | None, tuple[str, ...]]:
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("safetensors index must contain a non-empty weight_map")
    if len(raw_map) > _MAX_TENSORS:
        raise ValueError(
            "safetensors weight_map exceeds inspector safety limit of "
            f"{_MAX_TENSORS} tensor entries"
        )
    weight_map: dict[str, str] = {}
    shard_names: set[str] = set()
    for tensor_name, shard in raw_map.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or len(tensor_name.encode("utf-8")) > _MAX_TENSOR_NAME_BYTES
            or not isinstance(shard, str)
        ):
            raise ValueError("safetensors weight_map keys and values must be strings")
        weight_map[tensor_name] = shard
        shard_names.add(shard)
        if len(shard_names) > _MAX_SHARDS:
            raise ValueError(
                "safetensors weight_map exceeds inspector safety limit of "
                f"{_MAX_SHARDS} shards"
            )
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors index metadata must be an object")
    total_size = metadata.get("total_size")
    if total_size is not None and (not _is_int(total_size) or total_size <= 0):
        raise ValueError("safetensors index metadata.total_size must be positive")
    return weight_map, total_size, tuple(sorted(shard_names))


def _tensor_shape(raw: object, *, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError(f"invalid shape for safetensors tensor: {tensor_name}")
    if any(not _is_int(dimension) or dimension < 0 for dimension in raw):
        raise ValueError(f"invalid shape for safetensors tensor: {tensor_name}")
    return tuple(raw)


def _tensor_offsets(raw: object, *, tensor_name: str) -> tuple[int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(not _is_int(value) for value in raw)
    ):
        raise ValueError(f"invalid data_offsets for safetensors tensor: {tensor_name}")
    start, end = raw
    if start < 0 or end < start:
        raise ValueError(f"invalid data_offsets for safetensors tensor: {tensor_name}")
    return start, end


def _read_shard_header(
    path: Path,
    *,
    filename: str,
    remaining_header_bytes: int,
    remaining_tensor_entries: int,
) -> _ShardHeader:
    file_size = path.stat().st_size
    if file_size < 10:
        raise ValueError(f"safetensors shard is too small: {filename}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header length: {filename}")
        header_size = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_size < 2 or header_size > _HEADER_MAX_BYTES:
            raise ValueError(
                f"safetensors header size is invalid for {filename}: {header_size}"
            )
        if header_size > remaining_header_bytes:
            raise ValueError(
                "aggregate safetensors header bytes exceed inspector safety limit "
                f"of {_MAX_AGGREGATE_HEADER_BYTES}"
            )
        if 8 + header_size > file_size:
            raise ValueError(f"truncated safetensors header: {filename}")
        header_bytes = handle.read(header_size)
    try:
        payload = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header JSON: {filename}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"safetensors header root must be an object: {filename}")
    tensor_entry_count = len(payload) - int("__metadata__" in payload)
    if tensor_entry_count > remaining_tensor_entries:
        raise ValueError(
            "aggregate safetensors tensor entries exceed inspector safety limit "
            f"of {_MAX_TENSORS}"
        )
    header_metadata = payload.pop("__metadata__", None)
    if header_metadata is not None and not isinstance(header_metadata, dict):
        raise ValueError(f"invalid safetensors __metadata__: {filename}")

    data_size = file_size - 8 - header_size
    tensors: dict[str, _TensorMetadata] = {}
    for tensor_name, raw_metadata in payload.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or len(tensor_name.encode("utf-8")) > _MAX_TENSOR_NAME_BYTES
            or not isinstance(raw_metadata, dict)
        ):
            raise ValueError(f"invalid safetensors tensor entry in {filename}")
        dtype = raw_metadata.get("dtype")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"unsupported safetensors dtype for {tensor_name}: {dtype}"
            )
        shape = _tensor_shape(raw_metadata.get("shape"), tensor_name=tensor_name)
        start, end = _tensor_offsets(
            raw_metadata.get("data_offsets"), tensor_name=tensor_name
        )
        logical_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - start != logical_bytes:
            raise ValueError(
                f"safetensors byte length does not match shape for {tensor_name}"
            )
        if end > data_size:
            raise ValueError(f"safetensors tensor exceeds shard: {tensor_name}")
        tensors[tensor_name] = _TensorMetadata(
            name=tensor_name,
            shard=filename,
            dtype=dtype,
            shape=shape,
            offset_start=start,
            offset_end=end,
            logical_bytes=logical_bytes,
        )

    ordered = sorted(tensors.values(), key=lambda tensor: tensor.offset_start)
    cursor = 0
    for tensor in ordered:
        if tensor.offset_start != cursor:
            raise ValueError(
                f"safetensors data offsets are not contiguous in {filename}: "
                f"{tensor.name} starts at {tensor.offset_start}, expected {cursor}"
            )
        cursor = tensor.offset_end
    if cursor != data_size:
        raise ValueError(
            f"safetensors data payload is not fully indexed in {filename}: "
            f"{cursor} != {data_size}"
        )
    return _ShardHeader(
        filename=filename,
        file_size_bytes=file_size,
        header_size_bytes=header_size,
        tensors=tensors,
    )


def _load_tensors(
    snapshot: Path,
    *,
    index_path: Path | None,
    single_path: Path | None,
) -> tuple[
    dict[str, _TensorMetadata],
    tuple[_ShardHeader, ...],
    int | None,
    str,
]:
    if index_path is not None:
        index = _read_json_object(
            index_path, name="safetensors index", max_bytes=_INDEX_MAX_BYTES
        )
        weight_map, declared_total, shard_names = _weight_map(index)
        headers_list: list[_ShardHeader] = []
        physical_shards: dict[tuple[int, int], str] = {}
        aggregate_header_bytes = 0
        aggregate_tensor_entries = 0
        for filename in shard_names:
            shard_path = _safe_shard_path(snapshot, filename)
            shard_stat = shard_path.stat()
            identity = (shard_stat.st_dev, shard_stat.st_ino)
            previous_name = physical_shards.get(identity)
            if previous_name is not None:
                raise ValueError(
                    "safetensors index aliases one physical shard under multiple "
                    f"names: {previous_name!r}, {filename!r}"
                )
            physical_shards[identity] = filename
            header = _read_shard_header(
                shard_path,
                filename=filename,
                remaining_header_bytes=(
                    _MAX_AGGREGATE_HEADER_BYTES - aggregate_header_bytes
                ),
                remaining_tensor_entries=(_MAX_TENSORS - aggregate_tensor_entries),
            )
            aggregate_header_bytes += header.header_size_bytes
            aggregate_tensor_entries += len(header.tensors)
            headers_list.append(header)
        headers = tuple(headers_list)
        by_shard = {header.filename: header for header in headers}
        assigned_by_shard: dict[str, set[str]] = {
            filename: set() for filename in shard_names
        }
        tensors: dict[str, _TensorMetadata] = {}
        for tensor_name, shard_name in weight_map.items():
            assigned_by_shard[shard_name].add(tensor_name)
            header = by_shard[shard_name]
            try:
                tensors[tensor_name] = header.tensors[tensor_name]
            except KeyError as exc:
                raise ValueError(
                    f"index tensor is absent from declared shard: "
                    f"{tensor_name} -> {shard_name}"
                ) from exc
        for header in headers:
            unexpected = set(header.tensors) - assigned_by_shard[header.filename]
            if unexpected:
                first = min(unexpected)
                raise ValueError(
                    "safetensors shard contains an unindexed or misplaced tensor: "
                    f"{first}"
                )
        return tensors, headers, declared_total, "huggingface_index"

    if single_path is None:  # pragma: no cover - guarded by _resolve_inputs
        raise RuntimeError("checkpoint input resolution lost the unsharded file")
    filename = single_path.name
    header = _read_shard_header(
        single_path,
        filename=filename,
        remaining_header_bytes=_MAX_AGGREGATE_HEADER_BYTES,
        remaining_tensor_entries=_MAX_TENSORS,
    )
    return dict(header.tensors), (header,), None, "unsharded_synthesized"


def _projection_shape(
    projection: str, *, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    if projection in ("gate_proj", "up_proj"):
        return (intermediate_size, hidden_size)
    return (hidden_size, intermediate_size)


def _expert_records(
    tensors: dict[str, _TensorMetadata],
) -> tuple[
    str,
    dict[tuple[int, int], dict[str, _TensorMetadata]],
    set[str],
]:
    schemes: set[str] = set()
    records: dict[tuple[int, int], dict[str, _TensorMetadata]] = {}
    matched_names: set[str] = set()
    for name, tensor in tensors.items():
        match = _CANONICAL_EXPERT.fullmatch(name)
        if match is not None:
            scheme = "gate_up_down"
            projection = match.group("projection")
        else:
            match = _LEGACY_MIXTRAL_EXPERT.fullmatch(name)
            if match is None:
                continue
            scheme = "mixtral_w1_w2_w3"
            projection = _LEGACY_PROJECTION_MAP[match.group("projection")]
        schemes.add(scheme)
        key = (int(match.group("layer")), int(match.group("expert")))
        projection_records = records.setdefault(key, {})
        if projection in projection_records:
            raise ValueError(
                f"duplicate logical expert projection at layer {key[0]}, "
                f"expert {key[1]}: {projection}"
            )
        projection_records[projection] = tensor
        matched_names.add(name)
    if not records:
        raise ValueError("checkpoint contains no supported per-expert weights")
    if len(schemes) != 1:
        raise ValueError("checkpoint mixes incompatible expert naming schemes")
    unknown = sorted(
        name for name in tensors if ".experts." in name and name not in matched_names
    )
    if unknown:
        raise ValueError(f"unsupported expert tensor naming: {unknown[0]}")
    return next(iter(schemes)), records, matched_names


def _validate_experts(
    model: ModelInspection,
    naming: str,
    records: dict[tuple[int, int], dict[str, _TensorMetadata]],
) -> ExpertInspection:
    observed_layers = tuple(sorted({layer for layer, _ in records}))
    if observed_layers != model.expected_expert_layers:
        raise ValueError(
            "expert layer coverage does not match config: "
            f"{observed_layers} != {model.expected_expert_layers}"
        )
    expected_ids = tuple(range(model.experts_per_layer))
    placements: list[ExpertPlacement] = []
    expert_dtype: str | None = None
    bytes_per_expert: int | None = None
    logical_bytes = 0
    tensor_count = 0
    for layer in observed_layers:
        observed_ids = tuple(
            sorted(expert for record_layer, expert in records if record_layer == layer)
        )
        if observed_ids != expected_ids:
            raise ValueError(
                f"expert coverage does not match config at layer {layer}: "
                f"{observed_ids} != {expected_ids}"
            )
        for expert in expected_ids:
            projections = records[(layer, expert)]
            if set(projections) != set(_CANONICAL_PROJECTIONS):
                missing = sorted(set(_CANONICAL_PROJECTIONS) - set(projections))
                raise ValueError(
                    f"incomplete expert at layer {layer}, expert {expert}: "
                    + ", ".join(missing)
                )
            current_dtype: str | None = None
            current_bytes = 0
            for projection in _CANONICAL_PROJECTIONS:
                tensor = projections[projection]
                expected_shape = _projection_shape(
                    projection,
                    hidden_size=model.hidden_size,
                    intermediate_size=model.expert_intermediate_size,
                )
                if tensor.shape != expected_shape:
                    raise ValueError(
                        f"expert shape mismatch at layer {layer}, expert {expert}, "
                        f"{projection}: {tensor.shape} != {expected_shape}"
                    )
                if current_dtype is None:
                    current_dtype = tensor.dtype
                elif current_dtype != tensor.dtype:
                    raise ValueError(
                        f"mixed expert dtypes at layer {layer}, expert {expert}"
                    )
                current_bytes += tensor.logical_bytes
            if expert_dtype is None:
                expert_dtype = current_dtype
            elif expert_dtype != current_dtype:
                raise ValueError("expert dtype changes across layers or experts")
            if bytes_per_expert is None:
                bytes_per_expert = current_bytes
            elif bytes_per_expert != current_bytes:
                raise ValueError("expert byte size changes across layers or experts")

            shards = tuple(sorted({tensor.shard for tensor in projections.values()}))
            colocated = len(shards) == 1
            ordered = sorted(projections.items(), key=lambda item: item[1].offset_start)
            contiguous = colocated and all(
                previous[1].offset_end == current[1].offset_start
                for previous, current in pairwise(ordered)
            )
            span_bytes = (
                ordered[-1][1].offset_end - ordered[0][1].offset_start
                if colocated
                else None
            )
            gap_bytes = span_bytes - current_bytes if span_bytes is not None else None
            placements.append(
                ExpertPlacement(
                    layer=layer,
                    expert=expert,
                    shards=shards,
                    colocated=colocated,
                    contiguous=contiguous,
                    projection_order=tuple(projection for projection, _ in ordered),
                    logical_bytes=current_bytes,
                    span_bytes=span_bytes,
                    gap_bytes=gap_bytes,
                )
            )
            logical_bytes += current_bytes
            tensor_count += len(projections)

    if expert_dtype is None or bytes_per_expert is None:  # pragma: no cover
        raise RuntimeError("validated expert inventory is unexpectedly empty")
    if model.declared_dtype is not None and expert_dtype != model.declared_dtype:
        raise ValueError(
            f"expert dtype does not match config: "
            f"{expert_dtype} != {model.declared_dtype}"
        )
    return ExpertInspection(
        naming=naming,
        layers=observed_layers,
        projections=_CANONICAL_PROJECTIONS,
        dtype=expert_dtype,
        hidden_size=model.hidden_size,
        intermediate_size=model.expert_intermediate_size,
        experts_per_layer=model.experts_per_layer,
        total_experts=len(placements),
        tensor_count=tensor_count,
        bytes_per_expert=bytes_per_expert,
        logical_bytes=logical_bytes,
        colocated_experts=sum(placement.colocated for placement in placements),
        contiguous_experts=sum(placement.contiguous for placement in placements),
        split_experts=sum(not placement.colocated for placement in placements),
        placements=tuple(placements),
    )


def _shared_records(
    tensors: dict[str, _TensorMetadata],
) -> tuple[
    dict[int, dict[str, _TensorMetadata]],
    dict[int, _TensorMetadata],
    set[str],
]:
    projections: dict[int, dict[str, _TensorMetadata]] = {}
    gates: dict[int, _TensorMetadata] = {}
    names: set[str] = set()
    for name, tensor in tensors.items():
        match = _SHARED_EXPERT.fullmatch(name)
        if match is not None:
            layer = int(match.group("layer"))
            projection = match.group("projection")
            layer_records = projections.setdefault(layer, {})
            if projection in layer_records:
                raise ValueError(
                    f"duplicate shared-expert projection at layer {layer}: {projection}"
                )
            layer_records[projection] = tensor
            names.add(name)
            continue
        match = _SHARED_EXPERT_GATE.fullmatch(name)
        if match is not None:
            layer = int(match.group("layer"))
            if layer in gates:
                raise ValueError(f"duplicate shared-expert gate at layer {layer}")
            gates[layer] = tensor
            names.add(name)
    return projections, gates, names


def _validate_shared_expert(
    model: ModelInspection,
    projections: dict[int, dict[str, _TensorMetadata]],
    gates: dict[int, _TensorMetadata],
) -> SharedExpertInspection:
    configured = model.shared_expert_intermediate_size
    if configured is None and not projections and not gates:
        return SharedExpertInspection(
            present=False,
            layers=(),
            dtype=None,
            hidden_size=model.hidden_size,
            intermediate_size=None,
            projection_tensor_count=0,
            gate_tensor_count=0,
            bytes_per_layer=None,
            logical_bytes=0,
            gate_logical_bytes=0,
        )
    if configured is None:
        raise ValueError(
            "checkpoint has shared-expert tensors but config has no "
            "shared_expert_intermediate_size"
        )
    observed_layers = tuple(sorted(projections))
    if observed_layers != model.expected_expert_layers:
        raise ValueError(
            "shared-expert layer coverage does not match config: "
            f"{observed_layers} != {model.expected_expert_layers}"
        )
    if tuple(sorted(gates)) != model.expected_expert_layers:
        raise ValueError("shared-expert router gate coverage does not match config")

    shared_dtype: str | None = None
    bytes_per_layer: int | None = None
    logical_bytes = 0
    gate_bytes = 0
    for layer in observed_layers:
        layer_records = projections[layer]
        if set(layer_records) != set(_CANONICAL_PROJECTIONS):
            raise ValueError(f"incomplete shared expert at layer {layer}")
        current_bytes = 0
        for projection in _CANONICAL_PROJECTIONS:
            tensor = layer_records[projection]
            expected_shape = _projection_shape(
                projection,
                hidden_size=model.hidden_size,
                intermediate_size=configured,
            )
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"shared-expert shape mismatch at layer {layer}, {projection}: "
                    f"{tensor.shape} != {expected_shape}"
                )
            if shared_dtype is None:
                shared_dtype = tensor.dtype
            elif shared_dtype != tensor.dtype:
                raise ValueError("shared-expert dtype changes across tensors")
            current_bytes += tensor.logical_bytes
        gate = gates[layer]
        if math.prod(gate.shape) != model.hidden_size:
            raise ValueError(
                f"shared-expert gate shape mismatch at layer {layer}: {gate.shape}"
            )
        if shared_dtype != gate.dtype:
            raise ValueError(f"shared-expert gate dtype mismatch at layer {layer}")
        gate_bytes += gate.logical_bytes
        if bytes_per_layer is None:
            bytes_per_layer = current_bytes
        elif bytes_per_layer != current_bytes:
            raise ValueError("shared-expert byte size changes across layers")
        logical_bytes += current_bytes
    if model.declared_dtype is not None and shared_dtype != model.declared_dtype:
        raise ValueError("shared-expert dtype does not match config")
    return SharedExpertInspection(
        present=True,
        layers=observed_layers,
        dtype=shared_dtype,
        hidden_size=model.hidden_size,
        intermediate_size=configured,
        projection_tensor_count=sum(len(records) for records in projections.values()),
        gate_tensor_count=len(gates),
        bytes_per_layer=bytes_per_layer,
        logical_bytes=logical_bytes,
        gate_logical_bytes=gate_bytes,
    )


def _warnings(
    snapshot: Path,
    config: dict[str, Any],
    *,
    index_kind: str,
    declared_total: int | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if index_kind == "unsharded_synthesized":
        warnings.append(
            "No safetensors index was present; the unsharded header was used as "
            "the tensor inventory."
        )
    if declared_total is None:
        warnings.append("The checkpoint index does not declare metadata.total_size.")
    if "auto_map" in config:
        warnings.append(
            "config.json contains auto_map; the inspector did not import or execute it."
        )
    executable = sorted(
        path.name
        for path in snapshot.iterdir()
        if path.is_file() and path.suffix.lower() in {".py", ".pt", ".pth", ".bin"}
    )
    if executable:
        warnings.append(
            "Executable or pickle-capable files were ignored: " + ", ".join(executable)
        )
    return tuple(warnings)


def inspect_checkpoint(
    snapshot_or_index: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> CheckpointInspection:
    """Inspect a local MoE safetensors snapshot without loading tensor data.

    Args:
        snapshot_or_index: Snapshot directory, ``model.safetensors.index.json``,
            or one unsharded ``.safetensors`` file.
        config_path: Optional explicit local config path.  Defaults to
            ``config.json`` beside the checkpoint.

    Returns:
        A frozen, JSON-ready :class:`CheckpointInspection`.

    Raises:
        FileNotFoundError: A required local input is missing.
        ValueError: JSON, safetensors structure, config coverage, names, shapes,
            dtypes, offsets, or shard placement metadata are inconsistent.
    """

    snapshot, config_file, index_path, single_path = _resolve_inputs(
        snapshot_or_index, config_path
    )
    config = _read_json_object(
        config_file, name="checkpoint config", max_bytes=_CONFIG_MAX_BYTES
    )
    model = _model_inspection(config)
    tensors, headers, declared_total, index_kind = _load_tensors(
        snapshot,
        index_path=index_path,
        single_path=single_path,
    )
    logical_bytes = sum(tensor.logical_bytes for tensor in tensors.values())
    if declared_total is not None and declared_total != logical_bytes:
        raise ValueError(
            "safetensors index metadata.total_size does not match tensor headers: "
            f"{declared_total} != {logical_bytes}"
        )

    naming, expert_records, expert_names = _expert_records(tensors)
    experts = _validate_experts(model, naming, expert_records)
    shared_records, shared_gates, shared_names = _shared_records(tensors)
    shared = _validate_shared_expert(model, shared_records, shared_gates)

    shard_reports = tuple(
        ShardInspection(
            filename=header.filename,
            file_size_bytes=header.file_size_bytes,
            header_size_bytes=header.header_size_bytes,
            tensor_count=len(header.tensors),
            logical_bytes=sum(
                tensor.logical_bytes for tensor in header.tensors.values()
            ),
            expert_tensor_count=sum(name in expert_names for name in header.tensors),
            shared_expert_tensor_count=sum(
                name in shared_names for name in header.tensors
            ),
        )
        for header in headers
    )
    return CheckpointInspection(
        schema="moevm.checkpoint-inspection",
        schema_version=1,
        snapshot_path=str(snapshot),
        config_path=str(config_file),
        index_path=str(index_path) if index_path is not None else None,
        index_kind=index_kind,
        read_only=True,
        network_used=False,
        model_code_executed=False,
        model=model,
        experts=experts,
        shared_expert=shared,
        shards=shard_reports,
        tensor_count=len(tensors),
        logical_tensor_bytes=logical_bytes,
        declared_total_size_bytes=declared_total,
        warnings=_warnings(
            snapshot,
            config,
            index_kind=index_kind,
            declared_total=declared_total,
        ),
    )


__all__ = [
    "CheckpointInspection",
    "ExpertInspection",
    "ExpertPlacement",
    "ModelInspection",
    "ShardInspection",
    "SharedExpertInspection",
    "inspect_checkpoint",
]
