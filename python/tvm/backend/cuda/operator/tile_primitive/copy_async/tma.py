# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""CUDA Tensor Memory Accelerator dispatches.

``tma_auto`` derives the largest hardware-legal TMA box from the shared
memory iteration order.  ``tma_explicit`` maps the user's global tensor,
layout, and region directly to one TensorMap and one TMA instruction.

Both planners construct :class:`TensorMapSpec` in CUDA Driver API order:
dimension zero is innermost and ``global_strides`` omits dimension zero.
Validation, descriptor caching/encoding, prefetch, and PTX emission are
shared after that boundary.
"""

import re
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise

import tvm
from tvm.arith import Analyzer
from tvm.script import tirx as T
from tvm.tirx import Buffer, IntImm, PrimFunc, is_buffer_var
from tvm.tirx.layout import Layout, TileLayout
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..exec_scope_utils import single_thread
from ..layout_utils import strip_swizzle_to_tile
from ..tma_utils import SwizzleMode, get_swizzle_mode_from_layout


class ProofStatus(Enum):
    """Static proof result used by the shared TensorMap validator."""

    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ValidationFinding:
    """One named TensorMap validation rule."""

    rule: str
    status: ProofStatus
    message: str
    repairable: bool = False


@dataclass(frozen=True)
class TensorMapSpec:
    """A complete TensorMap plus instruction contract in CUDA API order."""

    descriptor_dtype: str
    descriptor_bits: int
    effective_bytes: int
    packed_kind: str | None
    force_cu_dtype: int
    base: object
    base_key: str
    descriptor_name: str
    base_byte_offset: object
    global_dims: tuple
    global_strides: tuple
    inner_stride: object
    box_dims: tuple
    element_strides: tuple
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int
    direction: str
    load_mode: str
    target_arch: str
    coordinates: tuple
    gather4: tuple
    smem_buffer: Buffer
    smem_start: tuple
    smem_base_offset: object
    mbar: object | None
    mbar_is_shared_addr: bool
    cta_group: int
    cta_mask: object
    cache_hint: object
    cache_policy: object
    use_tma_reduce: object | None
    payload_bits: object
    transaction_bits: object

    @property
    def rank(self) -> int:
        return len(self.global_dims)

    @property
    def is_packed(self) -> bool:
        return self.packed_kind is not None


@dataclass(frozen=True)
class IssueCoord:
    """One mixed-radix contribution from an auto issue axis."""

    dim_idx: int
    divisor: object
    modulus: object


@dataclass(frozen=True)
class AutoIssueAxis:
    """One unboxed shared-memory iterator lowered as an issue loop."""

    extent: object
    smem_stride: object
    coords: tuple[IssueCoord, ...]


@dataclass(frozen=True)
class TMAPlan:
    """A validated spec plus optional auto issue loops."""

    spec: TensorMapSpec
    issue_axes: tuple[AutoIssueAxis, ...] = ()
    shared_layout: TileLayout | None = None

    def issue_extent(self):
        total = 1
        for axis in self.issue_axes:
            total = total * axis.extent
        return total

    def offsets_and_coords(self, loop_var):
        """Return shared offset and CUDA-order coordinates for one issue."""

        cumulative = 1
        divisors = [None] * len(self.issue_axes)
        for idx in range(len(self.issue_axes) - 1, -1, -1):
            divisors[idx] = cumulative
            cumulative = cumulative * self.issue_axes[idx].extent

        smem_offset = 0
        coordinates = list(self.spec.coordinates)
        for axis, flat_divisor in zip(self.issue_axes, divisors):
            value = tvm.tirx.floormod(tvm.tirx.floordiv(loop_var, flat_divisor), axis.extent)
            smem_offset = smem_offset + value * axis.smem_stride
            for contribution in axis.coords:
                digit = tvm.tirx.floormod(
                    tvm.tirx.floordiv(value, contribution.divisor),
                    contribution.modulus,
                )
                coordinates[contribution.dim_idx] = coordinates[contribution.dim_idx] + digit
        return smem_offset, coordinates


@dataclass(frozen=True)
class _Gt:
    """One global-layout iterator after the two grouping steps."""

    extent: object
    stride: object
    smem_idx: int | None
    copy_dim: int
    global_dim: object
    coordinate: object


_COMMON_CONFIG = {
    "cache_hint",
    "cta_group",
    "cta_mask",
    "mbar",
    "mbarrier_addr",
    "oob",
    "prefetch_tensormap",
    "tensormap_l2_promotion",
    "tma_dtype",
    "use_tma_reduce",
}
_EXPLICIT_CONFIG = _COMMON_CONFIG | {"gather4", "src_selector"}
_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}
_PROMOTE_DTYPE = {1: ("uint16", 16), 2: ("uint32", 32), 4: ("uint64", 64)}
_REPAIRABLE_RULES = {
    "rank",
    "global_stride_alignment",
    "box_dim",
    "inner_box_bytes",
}
# These bounds depend only on runtime tensor arguments and are checked by
# runtime.cuTensorMapEncodeTiled before the CUDA driver encodes the descriptor.
_RUNTIME_VALIDATED_RULES = {"global_dim"}


def _proof(predicate, analyzer: Analyzer) -> ProofStatus:
    if isinstance(predicate, bool):
        return ProofStatus.PROVEN if predicate else ProofStatus.DISPROVEN
    if analyzer.can_prove(predicate):
        return ProofStatus.PROVEN
    if analyzer.can_prove(predicate == 0):
        return ProofStatus.DISPROVEN
    return ProofStatus.UNKNOWN


def _proof_all(analyzer: Analyzer, *conditions) -> ProofStatus:
    statuses = [_proof(condition, analyzer) for condition in conditions]
    if ProofStatus.DISPROVEN in statuses:
        return ProofStatus.DISPROVEN
    if ProofStatus.UNKNOWN in statuses:
        return ProofStatus.UNKNOWN
    return ProofStatus.PROVEN


def _proof_equal(lhs, rhs, analyzer: Analyzer) -> ProofStatus:
    if analyzer.can_prove_equal(lhs, rhs):
        return ProofStatus.PROVEN
    return _proof(lhs == rhs, analyzer)


def _require_proven(predicate, analyzer: Analyzer, stage: str, detail: str) -> None:
    status = _proof(predicate, analyzer)
    if status != ProofStatus.PROVEN:
        _auto_fail(stage, f"{detail} ({status.value})")


def _auto_fail(stage: str, detail: str):
    fail(
        f'tma_auto stage={stage}: {detail}; use dispatch="tma_explicit" '
        "when the mapping or hardware legality is only known at runtime"
    )


def _to_tile_layout(layout: Layout, shape) -> TileLayout:
    tile = strip_swizzle_to_tile(layout, lambda: list(shape))
    if not isinstance(tile, TileLayout):
        raise ValueError(f"expected TileLayout after removing swizzle, got {type(tile).__name__}")
    return tile


def _assert_plain_memory_layout(layout: TileLayout, label: str) -> None:
    if layout.replica:
        raise ValueError(f"{label} layout contains replica iterators")
    for iterator in layout.shard:
        if not iterator.axis.is_memory():
            raise ValueError(
                f"{label} layout must contain only memory iterators; got axis "
                f"{iterator.axis.name!r}"
            )
    for axis, offset in layout.offset.items():
        if not axis.is_memory() and not Analyzer().can_prove_equal(offset, 0):
            raise ValueError(f"{label} layout has non-memory offset {axis.name}={offset}")


def _layout_offset(layout: TileLayout):
    value = 0
    for axis, offset in layout.offset.items():
        if axis.is_memory():
            value = value + offset
    return Analyzer().simplify(value)


def _slice_layout(buffer: Buffer, starts, extents, label: str) -> tuple[TileLayout, TileLayout]:
    tile = _to_tile_layout(buffer.layout, buffer.shape)
    _assert_plain_memory_layout(tile, label)
    region = [(start, start + extent) for start, extent in zip(starts, extents)]
    sliced = tile.slice(list(buffer.shape), region)
    if sliced is None:
        raise ValueError(
            f"{label} layout cannot be sliced at start={list(starts)}, extent={list(extents)}"
        )
    if not isinstance(sliced, TileLayout):
        raise ValueError(f"{label} sliced layout is not a TileLayout")
    _assert_plain_memory_layout(sliced, f"sliced {label}")
    return tile, sliced


def _slice_global_layout(buffer: Buffer, starts, extents) -> tuple[TileLayout, TileLayout]:
    """Slice each logical global dimension without fusing across its boundary.

    ``TileLayout.slice`` canonicalizes the complete layout before grouping it
    by ``buffer.shape``.  For an unsigned dynamic shape, that can turn
    ``(n * C, K)`` into a single wrapping ``uint32`` product, after which the
    analyzer correctly refuses to prove the original dimension boundary.
    Global TensorMap planning needs those semantic buffer boundaries, so group
    the original memory layout first and slice each proven group separately.
    """

    analyzer = Analyzer()
    tile = _to_tile_layout(buffer.layout, buffer.shape)
    _assert_plain_memory_layout(tile, "global")
    try:
        grouped, separators = tile.group(list(buffer.shape))
    except (TypeError, ValueError, tvm.error.InternalError) as error:
        _auto_fail("global-slice", f"cannot group global layout by buffer shape: {error}")

    sliced_shard = []
    sliced_offset = dict(grouped.offset.items())
    for dim, (start, extent) in enumerate(zip(starts, extents)):
        group = TileLayout.from_iters(
            grouped.shard[separators[dim] : separators[dim + 1]],
        )
        sliced = group.slice([buffer.shape[dim]], [(start, start + extent)])
        if sliced is None or not isinstance(sliced, TileLayout):
            _auto_fail(
                "global-slice",
                f"global dimension {dim} cannot be sliced at start={start}, extent={extent}",
            )
        sliced_shard.extend(sliced.shard)
        for axis, value in sliced.offset.items():
            sliced_offset[axis] = analyzer.simplify(sliced_offset.get(axis, 0) + value)

    result = TileLayout.from_iters(sliced_shard, grouped.replica, sliced_offset)
    _assert_plain_memory_layout(result, "sliced global")
    return tile, result


def _target_sm(arch: str) -> int:
    match = re.search(r"sm_(\d+)", arch or "")
    return int(match.group(1)) if match else 0


def _normalize_l2_promotion(value) -> int:
    if value is None:
        return 2
    if isinstance(value, IntImm):
        value = int(value)
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        fail("TensorMap L2 promotion integer must be in [0, 3]")
    names = {
        "none": 0,
        "L2::none": 0,
        "L2::64B": 1,
        "L2::128B": 2,
        "L2::256B": 3,
    }
    if value in names:
        return names[value]
    fail("TensorMap L2 promotion must be None, 0..3, 'none', 'L2::64B', 'L2::128B', or 'L2::256B'")


def _normalize_oob(value) -> int:
    if value is None or value == "zero":
        return 0
    if value == "nan":
        return 1
    fail(f"unsupported TensorMap oob={value!r}; expected None, 'zero', or 'nan'")


def _normalize_cache_hint(cache_hint):
    if cache_hint is None:
        return "", None
    if isinstance(cache_hint, str):
        return cache_hint, None
    if isinstance(cache_hint, tvm.tirx.Expr):
        return "", cache_hint
    fail(f"cache_hint must be a string or TIR expression, got {type(cache_hint).__name__}")


def _dtype_contract(dtype, tma_dtype=None):
    data_type = tvm.DataType(dtype)
    name = str(data_type)
    if data_type.lanes != 1:
        fail(f"TensorMap descriptor dtype must have lanes=1, got {data_type}")

    valid = {
        "int8",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "float4_e2m1fn",
    }
    if name not in valid:
        fail(f"unsupported TensorMap descriptor dtype {name}")

    force_cu_dtype = -1
    if tma_dtype is not None:
        if tma_dtype not in ("tf32", "tfloat32"):
            fail("tma_dtype must be 'tf32' or 'tfloat32'")
        if name != "float32":
            fail(f"tma_dtype={tma_dtype!r} requires a float32 global buffer, got {name}")
        force_cu_dtype = 11

    bits = data_type.bits
    packed_kind = "16u4_align16" if name == "float4_e2m1fn" else None
    return name, bits, (bits + 7) // 8, packed_kind, force_cu_dtype


def _elements_to_bytes(
    elements, bits: int, analyzer: Analyzer, *, auto: bool, stage: str, label: str
):
    total_bits = analyzer.simplify(elements * bits)
    divisible = _proof_equal(tvm.tirx.floormod(total_bits, 8), 0, analyzer)
    if divisible == ProofStatus.DISPROVEN or (auto and divisible == ProofStatus.UNKNOWN):
        message = f"{label}={elements} elements at {bits} bits is not provably byte aligned"
        if auto:
            _auto_fail(stage, message)
        fail(f"tma_explicit stage={stage}: {message}")
    return analyzer.simplify(tvm.tirx.floordiv(total_bits, 8))


def _buffer_base(buffer: Buffer, *, auto: bool, stage: str):
    analyzer = Analyzer()
    layout = _to_tile_layout(buffer.layout, buffer.shape)
    layout_offset = _layout_offset(layout)
    element_offset = analyzer.simplify(buffer.elem_offset + layout_offset)
    byte_offset = _elements_to_bytes(
        element_offset,
        tvm.DataType(buffer.dtype).bits,
        analyzer,
        auto=auto,
        stage=stage,
        label="view base offset",
    )
    base = tvm.tirx.handle_add_byte_offset(buffer.data, byte_offset)
    return base, f"{hash(buffer.data)}:{base}", byte_offset


def _finding(
    findings: list[ValidationFinding],
    rule: str,
    status: ProofStatus,
    message: str,
    repairable: bool = False,
):
    findings.append(ValidationFinding(rule, status, message, repairable))


def validate_tensor_map_spec(spec: TensorMapSpec) -> tuple[ValidationFinding, ...]:
    """Validate one TensorMap using the static half of the runtime rule matrix."""

    analyzer = Analyzer()
    findings: list[ValidationFinding] = []
    dtype_bits = {
        "int8": 8,
        "int32": 32,
        "int64": 64,
        "uint8": 8,
        "uint16": 16,
        "uint32": 32,
        "uint64": 64,
        "float16": 16,
        "float32": 32,
        "float64": 64,
        "bfloat16": 16,
        "float8_e4m3fn": 8,
        "float8_e5m2": 8,
        "float4_e2m1fn": 4,
    }

    rank_ok = _proof(1 <= spec.rank <= 5, analyzer)
    _finding(
        findings,
        "rank",
        rank_ok,
        f"descriptor rank must be in [1, 5], got {spec.rank}",
        repairable=True,
    )

    if spec.descriptor_dtype not in dtype_bits:
        _finding(
            findings,
            "dtype",
            ProofStatus.DISPROVEN,
            f"unsupported descriptor dtype {spec.descriptor_dtype}",
        )
    else:
        _finding(findings, "dtype", ProofStatus.PROVEN, "descriptor dtype is supported")
        _finding(
            findings,
            "dtype_bits",
            _proof(spec.descriptor_bits == dtype_bits[spec.descriptor_dtype], analyzer),
            f"descriptor dtype {spec.descriptor_dtype} requires "
            f"{dtype_bits[spec.descriptor_dtype]} bits, got {spec.descriptor_bits}",
        )
        _finding(
            findings,
            "effective_bytes",
            _proof(spec.effective_bytes == (spec.descriptor_bits + 7) // 8, analyzer),
            f"effective bytes {spec.effective_bytes} do not match "
            f"{spec.descriptor_bits}-bit descriptor units",
        )

    expected_packed = "16u4_align16" if spec.descriptor_dtype == "float4_e2m1fn" else None
    _finding(
        findings,
        "packed_kind",
        _proof(spec.packed_kind == expected_packed, analyzer),
        f"descriptor dtype {spec.descriptor_dtype} requires packed_kind={expected_packed!r}, "
        f"got {spec.packed_kind!r}",
    )
    force_dtype_ok = spec.force_cu_dtype == -1 or (
        spec.force_cu_dtype == 11
        and spec.descriptor_dtype == "float32"
        and spec.descriptor_bits == 32
    )
    _finding(
        findings,
        "forced_cuda_dtype",
        _proof(force_dtype_ok, analyzer),
        "forced CUDA dtype must be -1, or TFLOAT32 (11) for a float32 descriptor",
    )

    array_lengths_ok = (
        len(spec.global_strides) == max(spec.rank - 1, 0)
        and len(spec.box_dims) == spec.rank
        and len(spec.element_strides) == spec.rank
        and len(spec.coordinates) == spec.rank
    )
    _finding(
        findings,
        "array_lengths",
        _proof(array_lengths_ok, analyzer),
        "TensorMap arrays must contain rank global dimensions, boxes, element strides, "
        "and coordinates, plus rank-1 byte strides",
    )

    required_alignment = 32 if spec.interleave == 2 or spec.packed_kind else 16
    base_alignment = _proof_equal(
        tvm.tirx.floormod(spec.base_byte_offset, required_alignment), 0, analyzer
    )
    _finding(
        findings,
        "global_base_alignment",
        base_alignment,
        f"view base byte offset {spec.base_byte_offset} must preserve "
        f"{required_alignment}B global alignment",
    )

    for idx, dim in enumerate(spec.global_dims):
        status = _proof_all(analyzer, dim > 0, dim <= (1 << 32))
        _finding(
            findings,
            "global_dim",
            status,
            f"globalDim[{idx}]={dim} must be in (0, 2^32]",
        )

    for idx, stride in enumerate(spec.global_strides):
        nonnegative = _proof(stride >= 0, analyzer)
        _finding(
            findings,
            "global_stride_range",
            nonnegative,
            f"globalStrides[{idx}]={stride} must be non-negative",
        )
        bounded = _proof(stride < (1 << 40), analyzer)
        _finding(
            findings,
            "global_stride_range",
            bounded,
            f"globalStrides[{idx}]={stride} must be less than 2^40",
        )
        alignment = 32 if spec.interleave == 2 or spec.packed_kind else 16
        aligned = _proof_equal(tvm.tirx.floormod(stride, alignment), 0, analyzer)
        _finding(
            findings,
            "global_stride_alignment",
            aligned,
            f"globalStrides[{idx}]={stride} must be a multiple of {alignment}B",
            repairable=True,
        )

    inner_stride = _proof_equal(spec.inner_stride, 1, analyzer)
    _finding(
        findings,
        "inner_stride",
        inner_stride,
        f"innermost global element stride must be 1, got {spec.inner_stride}",
    )

    for idx, box in enumerate(spec.box_dims):
        status = _proof_all(analyzer, box >= 1, box <= 256)
        _finding(
            findings,
            "box_dim",
            status,
            f"boxDim[{idx}]={box} must be in [1, 256]",
            repairable=True,
        )
    for idx, stride in enumerate(spec.element_strides):
        status = _proof_all(analyzer, stride >= 1, stride <= 8)
        _finding(
            findings,
            "element_stride",
            status,
            f"elementStride[{idx}]={stride} must be in [1, 8]",
        )
    if spec.element_strides:
        _finding(
            findings,
            "inner_stride",
            _proof_equal(spec.element_strides[0], 1, analyzer),
            f"innermost elementStride must be 1, got {spec.element_strides[0]}",
        )

    _finding(
        findings,
        "interleave",
        _proof(spec.interleave in (0, 1, 2), analyzer),
        f"interleave enum {spec.interleave} is invalid",
    )
    _finding(
        findings,
        "swizzle",
        _proof(spec.swizzle in (0, 1, 2, 3), analyzer),
        f"swizzle enum {spec.swizzle} is invalid",
    )
    _finding(
        findings,
        "l2_promotion",
        _proof(spec.l2_promotion in (0, 1, 2, 3), analyzer),
        f"L2 promotion enum {spec.l2_promotion} is invalid",
    )
    _finding(
        findings,
        "oob",
        _proof(spec.oob_fill in (0, 1), analyzer),
        f"OOB enum {spec.oob_fill} is invalid",
    )

    if spec.interleave != 0:
        _finding(
            findings,
            "interleave_rank",
            _proof(spec.rank >= 3, analyzer),
            "interleaved TensorMaps require rank >= 3",
        )
    if spec.interleave == 2:
        _finding(
            findings,
            "interleave_swizzle",
            _proof(spec.swizzle == 1, analyzer),
            "32B interleave requires the 32B swizzle mode",
        )

    if spec.box_dims and spec.interleave == 0 and not spec.is_packed:
        inner_bytes = analyzer.simplify(spec.box_dims[0] * spec.effective_bytes)
        _finding(
            findings,
            "inner_box_bytes",
            _proof_equal(tvm.tirx.floormod(inner_bytes, 16), 0, analyzer),
            f"innermost box is {inner_bytes}B; non-interleaved TensorMaps require a 16B multiple",
            repairable=True,
        )
        atom_bytes = {1: 32, 2: 64, 3: 128}.get(spec.swizzle)
        if atom_bytes is not None:
            _finding(
                findings,
                "swizzle_inner_box",
                _proof(inner_bytes <= atom_bytes, analyzer),
                f"innermost box {inner_bytes}B exceeds the {atom_bytes}B swizzle atom",
            )

    if spec.packed_kind == "16u4_align16" and spec.rank and spec.box_dims:
        _finding(
            findings,
            "packed_shape",
            _proof_equal(tvm.tirx.floormod(spec.global_dims[0], 128), 0, analyzer),
            "packed 16U4 align16 requires globalDim[0] to be a multiple of 128",
        )
        _finding(
            findings,
            "packed_box",
            _proof_equal(spec.box_dims[0], 128, analyzer),
            "packed 16U4 align16 requires boxDim[0] == 128",
        )
        _finding(
            findings,
            "packed_swizzle",
            _proof(spec.swizzle in (0, 3), analyzer),
            "packed 16U4 align16 supports only NONE or 128B swizzle in this layout model",
        )
        _finding(
            findings,
            "packed_direction",
            _proof(spec.direction == "g2s", analyzer),
            "packed 16U4 align16 TensorMaps are load-only",
        )

    if spec.oob_fill == 1:
        floating = spec.descriptor_dtype in _FLOAT_DTYPES or spec.force_cu_dtype == 11
        _finding(
            findings,
            "nan_oob_dtype",
            _proof(floating and not spec.is_packed, analyzer),
            "NaN OOB fill requires a non-packed floating-point descriptor",
        )

    if spec.direction not in ("g2s", "s2g"):
        _finding(
            findings,
            "direction",
            ProofStatus.DISPROVEN,
            f"invalid TMA direction {spec.direction!r}",
        )
    if spec.load_mode == "tile_gather4":
        _finding(
            findings,
            "gather4_rank",
            _proof(spec.rank == 2, analyzer),
            f"gather4 requires a rank-2 TensorMap, got rank {spec.rank}",
        )
        _finding(
            findings,
            "gather4_rows",
            _proof(len(spec.gather4) == 4, analyzer),
            f"gather4 requires exactly four row coordinates, got {len(spec.gather4)}",
        )
        _finding(
            findings,
            "gather4_box",
            _proof_equal(spec.box_dims[1], 1, analyzer)
            if spec.rank == 2 and len(spec.box_dims) == 2
            else ProofStatus.DISPROVEN,
            "gather4 public axis 0 must map to a hardware row box of one",
        )
        _finding(
            findings,
            "gather4_interleave",
            _proof(spec.interleave == 0, analyzer),
            "gather4 does not support interleaved TensorMaps",
        )
        _finding(
            findings,
            "target",
            _proof(_target_sm(spec.target_arch) >= 100, analyzer),
            f"gather4 requires SM100+, got {spec.target_arch!r}",
        )
    else:
        _finding(
            findings,
            "coordinate_count",
            _proof(len(spec.coordinates) == spec.rank, analyzer),
            f"coordinate count {len(spec.coordinates)} must equal rank {spec.rank}",
        )
        _finding(
            findings,
            "target",
            _proof(_target_sm(spec.target_arch) >= 90, analyzer),
            f"TMA requires SM90+, got {spec.target_arch!r}",
        )

    _finding(
        findings,
        "cta_group",
        _proof(spec.cta_group in (1, 2), analyzer),
        f"cta_group must be 1 or 2, got {spec.cta_group}",
    )
    if spec.cta_group == 2:
        _finding(
            findings,
            "target",
            _proof(_target_sm(spec.target_arch) >= 100, analyzer),
            f"cta_group=2 requires SM100+, got {spec.target_arch!r}",
        )
        _finding(
            findings,
            "mbarrier_address",
            _proof(spec.mbar_is_shared_addr, analyzer),
            "cta_group=2 requires a precomputed shared mbarrier address",
        )
    if spec.direction == "g2s":
        _finding(
            findings,
            "mbar",
            _proof(spec.mbar is not None, analyzer),
            "global-to-shared TMA requires mbar",
        )
        _finding(
            findings,
            "reduce_direction",
            _proof(spec.use_tma_reduce is None, analyzer),
            "TMA reduction is only valid for shared-to-global",
        )
    else:
        _finding(
            findings,
            "gather_direction",
            _proof(spec.load_mode == "tile", analyzer),
            "gather4 is only valid for global-to-shared",
        )
        _finding(
            findings,
            "cta_mask_direction",
            _proof(
                (isinstance(spec.cta_mask, int) and spec.cta_mask == 0)
                or (isinstance(spec.cta_mask, IntImm) and int(spec.cta_mask) == 0),
                analyzer,
            ),
            "cta_mask is only valid for global-to-shared",
        )

    return tuple(findings)


def _validation_failures(spec: TensorMapSpec, *, auto: bool):
    findings = validate_tensor_map_spec(spec)
    return [
        finding
        for finding in findings
        if finding.status == ProofStatus.DISPROVEN
        or (
            auto
            and finding.status == ProofStatus.UNKNOWN
            and finding.rule not in _RUNTIME_VALIDATED_RULES
        )
    ]


def _raise_validation(stage: str, failures, *, auto: bool):
    details = "; ".join(
        f"{finding.rule}: {finding.message} [{finding.status.value}]" for finding in failures
    )
    if auto:
        _auto_fail(stage, details)
    fail(f"tma_explicit stage={stage}: {details}")


def _validate_explicit(spec: TensorMapSpec, stage: str) -> None:
    failures = _validation_failures(spec, auto=False)
    if failures:
        _raise_validation(stage, failures, auto=False)


def _simplify_with_ranges(exprs, var_ranges, sctx: DispatchContext):
    analyzer = Analyzer()
    for var, extent in var_ranges:
        analyzer.bind(var, tvm.ir.Range.from_min_extent(0, extent))
    for var, value in sctx.var_range_map.items():
        if var not in [item[0] for item in var_ranges]:
            analyzer.bind(var, value)
    return [analyzer.simplify(expr) for expr in exprs]


def _smem_iter_order(sliced: TileLayout):
    analyzer = Analyzer()
    active = [
        idx
        for idx, iterator in enumerate(sliced.shard)
        if not analyzer.can_prove_equal(iterator.extent, 1)
    ]
    constant_strides = []
    for idx in active:
        stride = analyzer.simplify(sliced.shard[idx].stride)
        if not isinstance(stride, IntImm):
            _auto_fail(
                "shared-chain",
                f"shared iter {idx} stride={stride} is symbolic, so physical order is unknown",
            )
        constant_strides.append((int(stride), idx))
    constant_strides.sort()
    order = [idx for _, idx in constant_strides]
    if not order:
        _auto_fail("shared-chain", "copy region has no non-unit shared memory iterator")

    previous = sliced.shard[order[0]]
    _require_proven(
        previous.stride == 1,
        analyzer,
        "shared-chain",
        f"smallest shared stride must be 1, got {previous.stride}",
    )
    for position, idx in enumerate(order[1:], start=1):
        current = sliced.shard[idx]
        expected = analyzer.simplify(previous.stride * previous.extent)
        _require_proven(
            current.stride == expected,
            analyzer,
            "shared-chain",
            f"shared iter {idx} at sorted position {position} has stride={current.stride}, "
            f"expected {expected} after extent={previous.extent}",
        )
        previous = current
    return order


def _copy_region_parts(region):
    return [item.min for item in region], [item.extent for item in region]


def _build_auto_gt(
    g_buf: Buffer,
    g_starts,
    g_extents,
    sliced_smem: TileLayout,
    sctx: DispatchContext,
):
    analyzer = Analyzer()
    for var, value in sctx.var_range_map.items():
        analyzer.bind(var, value)
    _, sliced_gmem = _slice_global_layout(g_buf, g_starts, g_extents)

    smem_shape = tuple(iterator.extent for iterator in sliced_smem.shard)
    if not smem_shape:
        _auto_fail("group-global", "sliced shared layout has no iterators")
    try:
        grouped, (smem_sep, copy_sep) = sliced_gmem.group_many((smem_shape, tuple(g_extents)))
    except (TypeError, ValueError, tvm.error.TVMError) as error:
        _auto_fail(
            "group-global",
            f"cannot jointly group global layout by shared shape={smem_shape} "
            f"and copy shape={tuple(g_extents)}: {error}",
        )
    _assert_plain_memory_layout(grouped, "grouped global")

    shard_to_smem = {}
    for smem_idx in range(len(smem_shape)):
        for shard_idx in range(smem_sep[smem_idx], smem_sep[smem_idx + 1]):
            shard_to_smem[shard_idx] = smem_idx

    records = [None] * len(grouped.shard)
    for copy_dim, (start_idx, end_idx) in enumerate(pairwise(copy_sep)):
        iterators = list(grouped.shard[start_idx:end_idx])
        product = 1
        for iterator in iterators:
            product = analyzer.simplify(product * iterator.extent)
        _require_proven(
            product == g_extents[copy_dim],
            analyzer,
            "copy-group",
            f"group={copy_dim} iterator product={product} does not equal "
            f"copy extent={g_extents[copy_dim]}",
        )
        if not iterators:
            _auto_fail(
                "copy-group",
                f"group={copy_dim} start={g_starts[copy_dim]} "
                f"extent={g_extents[copy_dim]} has no global iterator",
            )

        suffix_products = [None] * len(iterators)
        suffix_product = 1
        for local_idx in range(len(iterators) - 1, -1, -1):
            suffix_products[local_idx] = suffix_product
            suffix_product = analyzer.simplify(suffix_product * iterators[local_idx].extent)
        inner_product = suffix_products[0]
        _require_proven(
            tvm.tirx.floormod(g_starts[copy_dim], inner_product) == 0,
            analyzer,
            "coordinate",
            f"group={copy_dim} start={g_starts[copy_dim]} is not divisible by "
            f"inner_product={inner_product}",
        )
        _require_proven(
            tvm.tirx.floormod(g_buf.shape[copy_dim], inner_product) == 0,
            analyzer,
            "global-shape",
            f"group={copy_dim} tensor extent={g_buf.shape[copy_dim]} is not divisible "
            f"by inner_product={inner_product}",
        )

        for local_idx, iterator in enumerate(iterators):
            shard_idx = start_idx + local_idx
            smem_idx = shard_to_smem.get(shard_idx)
            if analyzer.can_prove_equal(iterator.extent, 1):
                # Unit copy dimensions remain real TensorMap dimensions, but
                # they do not contribute shared-memory payload.
                smem_idx = None
            elif smem_idx is None:
                _require_proven(
                    iterator.extent == 1,
                    analyzer,
                    "coordinate-only-dim",
                    f"global iterator={shard_idx} copy group={copy_dim} extent={iterator.extent} "
                    "has no shared-memory iterator",
                )
            if local_idx == 0:
                global_dim = analyzer.simplify(
                    tvm.tirx.floordiv(g_buf.shape[copy_dim], inner_product)
                )
                coordinate = analyzer.simplify(tvm.tirx.floordiv(g_starts[copy_dim], inner_product))
            else:
                global_dim = iterator.extent
                coordinate = 0
            records[shard_idx] = _Gt(
                extent=iterator.extent,
                stride=iterator.stride,
                smem_idx=smem_idx,
                copy_dim=copy_dim,
                global_dim=global_dim,
                coordinate=coordinate,
            )

    if any(record is None for record in records):
        _auto_fail("copy-group", "global iterator was not assigned to a copy-region dimension")
    return tuple(records)


def _auto_spec_for_prefix(
    *,
    op_call,
    sctx,
    direction,
    s_buf,
    g_buf,
    s_starts,
    smem_layout_offset,
    sliced_smem,
    smem_order,
    gt_records,
    prefix,
    swizzle,
    descriptor,
    runtime,
):
    analyzer = Analyzer()
    selected_smem = set(smem_order[:prefix])
    gt_by_smem = {idx: [] for idx in range(len(sliced_smem.shard))}
    detached_gt = []
    for gt_idx, gt in enumerate(gt_records):
        if gt.smem_idx is None:
            detached_gt.append(gt_idx)
        else:
            gt_by_smem[gt.smem_idx].append((gt_idx, gt))

    cuda_gt_indices = []
    for smem_idx in smem_order:
        cuda_gt_indices.extend(idx for idx, _ in reversed(gt_by_smem[smem_idx]))
    unit_smem = [
        idx
        for idx, iterator in enumerate(sliced_smem.shard)
        if analyzer.can_prove_equal(iterator.extent, 1)
    ]
    for smem_idx in unit_smem:
        cuda_gt_indices.extend(idx for idx, _ in reversed(gt_by_smem[smem_idx]))
    cuda_gt_indices.extend(detached_gt)
    if len(cuda_gt_indices) != len(gt_records):
        _auto_fail(
            "descriptor-order",
            "not every global iterator maps to shared data or a coordinate-only dimension",
        )

    gt_to_dim = {gt_idx: dim_idx for dim_idx, gt_idx in enumerate(cuda_gt_indices)}
    ordered = [gt_records[idx] for idx in cuda_gt_indices]
    global_dims = tuple(gt.global_dim for gt in ordered)
    coordinates = tuple(gt.coordinate for gt in ordered)
    box_dims = tuple(gt.extent if gt.smem_idx in selected_smem else 1 for gt in ordered)

    descriptor_dtype, descriptor_bits, effective_bytes, packed_kind, force_cu_dtype = descriptor
    full_byte_strides = [
        _elements_to_bytes(
            gt.stride,
            descriptor_bits,
            analyzer,
            auto=True,
            stage="descriptor-stride",
            label=f"global stride for copy group {gt.copy_dim}",
        )
        for gt in ordered
    ]
    inner_stride = ordered[0].stride
    global_strides = tuple(full_byte_strides[1:])

    issue_axes = []
    for smem_idx in smem_order[prefix:]:
        iterator = sliced_smem.shard[smem_idx]
        group = gt_by_smem[smem_idx]
        group_product = 1
        for _, gt in group:
            group_product = analyzer.simplify(group_product * gt.extent)
        _require_proven(
            group_product == iterator.extent,
            analyzer,
            "issue-axis",
            f"shared group={smem_idx} extent={iterator.extent} maps to "
            f"global product={group_product}",
        )
        contributions = []
        inner_product = 1
        local = []
        for gt_idx, gt in reversed(group):
            local.append(
                IssueCoord(
                    dim_idx=gt_to_dim[gt_idx],
                    divisor=inner_product,
                    modulus=gt.extent,
                )
            )
            inner_product = analyzer.simplify(inner_product * gt.extent)
        contributions.extend(reversed(local))
        issue_axes.append(
            AutoIssueAxis(
                extent=iterator.extent,
                smem_stride=iterator.stride,
                coords=tuple(contributions),
            )
        )

    issue_count = 1
    for axis in issue_axes:
        issue_count = analyzer.simplify(issue_count * axis.extent)
    box_elements = 1
    for box in box_dims:
        box_elements = analyzer.simplify(box_elements * box)
    transaction_bits = analyzer.simplify(box_elements * descriptor_bits)
    payload_bits = analyzer.simplify(transaction_bits * issue_count)

    s_elements = 1
    for iterator in sliced_smem.shard:
        s_elements = analyzer.simplify(s_elements * iterator.extent)
    expected_bits = analyzer.simplify(s_elements * tvm.DataType(s_buf.dtype).bits)
    _require_proven(
        payload_bits == expected_bits,
        analyzer,
        "byte-equivalence",
        f"prefix={prefix} payload={payload_bits} bits, expected={expected_bits} bits",
    )

    base, base_key, base_byte_offset = _buffer_base(g_buf, auto=True, stage="global-base")
    spec = TensorMapSpec(
        descriptor_dtype=descriptor_dtype,
        descriptor_bits=descriptor_bits,
        effective_bytes=effective_bytes,
        packed_kind=packed_kind,
        force_cu_dtype=force_cu_dtype,
        base=base,
        base_key=base_key,
        descriptor_name=g_buf.name,
        base_byte_offset=base_byte_offset,
        global_dims=global_dims,
        global_strides=global_strides,
        inner_stride=inner_stride,
        box_dims=box_dims,
        element_strides=(1,) * len(global_dims),
        interleave=0,
        swizzle=swizzle.value,
        l2_promotion=runtime["l2_promotion"],
        oob_fill=runtime["oob_fill"],
        direction=direction,
        load_mode="tile",
        target_arch=sctx.target.arch,
        coordinates=coordinates,
        gather4=(),
        smem_buffer=s_buf,
        smem_start=tuple(s_starts),
        smem_base_offset=analyzer.simplify(s_buf.elem_offset + smem_layout_offset),
        mbar=runtime["mbar"],
        mbar_is_shared_addr=runtime["mbar_is_shared_addr"],
        cta_group=runtime["cta_group"],
        cta_mask=runtime["cta_mask"],
        cache_hint=runtime["cache_hint"],
        cache_policy=runtime["cache_policy"],
        use_tma_reduce=runtime["use_tma_reduce"],
        payload_bits=payload_bits,
        transaction_bits=transaction_bits,
    )
    return TMAPlan(
        spec=spec,
        issue_axes=tuple(issue_axes),
        shared_layout=_to_tile_layout(s_buf.layout, s_buf.shape),
    )


def _validate_auto_shared_mapping(
    plan: TMAPlan,
    sliced_smem: TileLayout,
    swizzle: SwizzleMode,
    sctx: DispatchContext,
) -> None:
    """Prove that issue loops partition the sliced shared region exactly once."""

    analyzer = Analyzer()
    for var, value in sctx.var_range_map.items():
        analyzer.bind(var, value)
    spec = plan.spec
    shared_bits = tvm.DataType(spec.smem_buffer.dtype).bits
    start_elements = spec.smem_base_offset
    start_bytes = _elements_to_bytes(
        start_elements,
        shared_bits,
        analyzer,
        auto=True,
        stage="shared-pointer",
        label="shared slice pointer",
    )
    atom_alignment = {0: 16, 1: 32, 2: 64, 3: 128}[swizzle.value]
    _require_proven(
        tvm.tirx.floormod(start_bytes, atom_alignment) == 0,
        analyzer,
        "shared-pointer",
        f"shared slice offset={start_bytes}B must preserve {atom_alignment}B alignment",
    )

    transaction_elements = analyzer.simplify(tvm.tirx.floordiv(spec.transaction_bits, shared_bits))
    _require_proven(
        tvm.tirx.floormod(spec.transaction_bits, shared_bits) == 0,
        analyzer,
        "issue-coverage",
        f"transaction={spec.transaction_bits} bits is not an integral number of "
        f"{shared_bits}-bit shared elements",
    )
    covered = transaction_elements
    for axis_idx, axis in enumerate(plan.issue_axes):
        _require_proven(
            axis.smem_stride == covered,
            analyzer,
            "issue-coverage",
            f"issue axis={axis_idx} stride={axis.smem_stride} does not follow "
            f"the covered prefix={covered}",
        )
        covered = analyzer.simplify(covered * axis.extent)

    sliced_elements = 1
    for iterator in sliced_smem.shard:
        sliced_elements = analyzer.simplify(sliced_elements * iterator.extent)
    _require_proven(
        covered == sliced_elements,
        analyzer,
        "issue-coverage",
        f"issue loops plus one box cover {covered} shared elements, "
        f"but the sliced region contains {sliced_elements}",
    )


def _remap_issue_axes_after_remove(issue_axes, removed_idx):
    remapped = []
    for axis in issue_axes:
        coords = []
        for coord in axis.coords:
            if coord.dim_idx == removed_idx:
                raise ValueError("attempted to remove an issue-driven descriptor dimension")
            coords.append(
                replace(
                    coord,
                    dim_idx=coord.dim_idx - 1 if coord.dim_idx > removed_idx else coord.dim_idx,
                )
            )
        remapped.append(replace(axis, coords=tuple(coords)))
    return tuple(remapped)


def _auto_inner_dimension_is_legal(spec, global_dim, box_dim, analyzer: Analyzer) -> bool:
    """Check rules that can change when an auto dimension becomes innermost."""

    if spec.interleave != 0:
        return True
    if spec.is_packed:
        checks = (
            _proof_equal(tvm.tirx.floormod(global_dim, 128), 0, analyzer),
            _proof_equal(box_dim, 128, analyzer),
        )
    else:
        inner_bytes = analyzer.simplify(box_dim * spec.effective_bytes)
        checks = [_proof_equal(tvm.tirx.floormod(inner_bytes, 16), 0, analyzer)]
        atom_bytes = {1: 32, 2: 64, 3: 128}.get(spec.swizzle)
        if atom_bytes is not None:
            checks.append(_proof(inner_bytes <= atom_bytes, analyzer))
    return all(status == ProofStatus.PROVEN for status in checks)


def _canonicalize_auto_plan(plan: TMAPlan) -> TMAPlan:
    """Canonicalize auto-only TensorMap dimensions without changing byte addresses."""

    analyzer = Analyzer()
    current = plan
    while True:
        spec = current.spec
        full_strides = [spec.effective_bytes * spec.inner_stride, *spec.global_strides]
        issue_dims = {coord.dim_idx for axis in current.issue_axes for coord in axis.coords}

        # A unit global dimension carries no address information for an
        # in-bounds tma_auto copy.  Non-innermost units can always disappear;
        # an innermost unit can disappear only when the next dimension already
        # has the implicit innermost byte stride.
        removed = False
        if spec.rank > 1:
            for dim_idx in range(spec.rank):
                if dim_idx in issue_dims:
                    continue
                checks = (
                    _proof_equal(spec.global_dims[dim_idx], 1, analyzer),
                    _proof_equal(spec.box_dims[dim_idx], 1, analyzer),
                    _proof_equal(spec.element_strides[dim_idx], 1, analyzer),
                )
                if any(status != ProofStatus.PROVEN for status in checks):
                    continue
                if dim_idx == 0:
                    if _proof_equal(
                        full_strides[1], full_strides[0], analyzer
                    ) != ProofStatus.PROVEN or not _auto_inner_dimension_is_legal(
                        spec,
                        spec.global_dims[1],
                        spec.box_dims[1],
                        analyzer,
                    ):
                        continue

                global_dims = list(spec.global_dims)
                box_dims = list(spec.box_dims)
                coordinates = list(spec.coordinates)
                element_strides = list(spec.element_strides)
                for values in (global_dims, box_dims, coordinates, element_strides):
                    values.pop(dim_idx)
                full_strides.pop(dim_idx)
                current = replace(
                    current,
                    spec=replace(
                        spec,
                        global_dims=tuple(global_dims),
                        global_strides=tuple(full_strides[1:]),
                        box_dims=tuple(box_dims),
                        coordinates=tuple(coordinates),
                        element_strides=tuple(element_strides),
                    ),
                    issue_axes=_remap_issue_axes_after_remove(current.issue_axes, dim_idx),
                )
                removed = True
                break
        if removed:
            continue

        # Flatten an adjacent pair when the inner dimension is copied in full
        # from coordinate zero and the outer byte stride follows it
        # contiguously.  The outer dimension may have a partial box and a
        # non-zero coordinate; both are scaled into the flattened dimension.
        merged = False
        promoted_for_merge = False
        for inner_idx in range(spec.rank - 1):
            outer_idx = inner_idx + 1
            if inner_idx in issue_dims or outer_idx in issue_dims:
                continue
            merged_global_dim = analyzer.simplify(
                spec.global_dims[inner_idx] * spec.global_dims[outer_idx]
            )
            merged_box_dim = analyzer.simplify(spec.box_dims[inner_idx] * spec.box_dims[outer_idx])
            checks = (
                _proof_equal(spec.box_dims[inner_idx], spec.global_dims[inner_idx], analyzer),
                _proof_equal(spec.coordinates[inner_idx], 0, analyzer),
                _proof_equal(
                    full_strides[outer_idx],
                    spec.global_dims[inner_idx] * full_strides[inner_idx],
                    analyzer,
                ),
                _proof_equal(spec.element_strides[inner_idx], 1, analyzer),
                _proof_equal(spec.element_strides[outer_idx], 1, analyzer),
                _proof_all(analyzer, merged_global_dim > 0, merged_global_dim <= (1 << 32)),
            )
            if inner_idx == 0 and not _auto_inner_dimension_is_legal(
                spec, merged_global_dim, merged_box_dim, analyzer
            ):
                continue
            if any(status != ProofStatus.PROVEN for status in checks):
                continue
            box_status = _proof_all(analyzer, merged_box_dim >= 1, merged_box_dim <= 256)
            if box_status != ProofStatus.PROVEN:
                # Preserve the innermost contiguous-chain boundary.  Skipping a
                # box-size-blocked inner merge and merging an outer pair instead
                # changes the TensorMap tiling even though one byte-preserving
                # descriptor-unit promotion may make this merge legal.
                if (
                    inner_idx == 0
                    and spec.rank > 2
                    and _proof(merged_box_dim > 256, analyzer) == ProofStatus.PROVEN
                    and _proof_equal(
                        spec.box_dims[outer_idx], spec.global_dims[outer_idx], analyzer
                    )
                    == ProofStatus.PROVEN
                    and _proof_equal(spec.coordinates[outer_idx], 0, analyzer) == ProofStatus.PROVEN
                ):
                    promoted = _promote_auto_once(current)
                    if promoted is not None:
                        promoted_spec = promoted.spec
                        promoted_global_dim = analyzer.simplify(
                            promoted_spec.global_dims[0] * promoted_spec.global_dims[1]
                        )
                        promoted_box_dim = analyzer.simplify(
                            promoted_spec.box_dims[0] * promoted_spec.box_dims[1]
                        )
                        if _proof_all(
                            analyzer,
                            promoted_box_dim >= 1,
                            promoted_box_dim <= 256,
                            promoted_global_dim > 0,
                            promoted_global_dim <= (1 << 32),
                        ) == ProofStatus.PROVEN and _auto_inner_dimension_is_legal(
                            promoted_spec,
                            promoted_global_dim,
                            promoted_box_dim,
                            analyzer,
                        ):
                            current = promoted
                            promoted_for_merge = True
                            break
                continue

            global_dims = list(spec.global_dims)
            box_dims = list(spec.box_dims)
            coordinates = list(spec.coordinates)
            element_strides = list(spec.element_strides)
            global_dims[inner_idx] = merged_global_dim
            box_dims[inner_idx] = merged_box_dim
            coordinates[inner_idx] = analyzer.simplify(
                spec.coordinates[outer_idx] * spec.global_dims[inner_idx]
            )
            for values in (global_dims, box_dims, coordinates, element_strides):
                values.pop(outer_idx)
            full_strides.pop(outer_idx)
            current = replace(
                current,
                spec=replace(
                    spec,
                    global_dims=tuple(global_dims),
                    global_strides=tuple(full_strides[1:]),
                    box_dims=tuple(box_dims),
                    coordinates=tuple(coordinates),
                    element_strides=tuple(element_strides),
                ),
                issue_axes=_remap_issue_axes_after_remove(current.issue_axes, outer_idx),
            )
            merged = True
            break
        if promoted_for_merge:
            continue
        if not merged:
            return current


def _promotion_allowed(plan: TMAPlan) -> bool:
    spec = plan.spec
    analyzer = Analyzer()
    if spec.effective_bytes not in _PROMOTE_DTYPE:
        return False
    if spec.is_packed or spec.interleave != 0 or spec.oob_fill != 0:
        return False
    if spec.load_mode != "tile" or spec.use_tma_reduce is not None:
        return False
    if spec.force_cu_dtype >= 0:
        return False
    if _proof_equal(spec.inner_stride, 1, analyzer) != ProofStatus.PROVEN:
        return False
    if any(
        _proof_equal(stride, 1, analyzer) != ProofStatus.PROVEN for stride in spec.element_strides
    ):
        return False
    return not any(coord.dim_idx == 0 for axis in plan.issue_axes for coord in axis.coords)


def _promote_auto_once(plan: TMAPlan) -> TMAPlan | None:
    """Promote descriptor units while preserving every byte address."""

    if not _promotion_allowed(plan):
        return None
    analyzer = Analyzer()
    spec = plan.spec
    new_dtype, new_bits = _PROMOTE_DTYPE[spec.effective_bytes]
    for value, label in (
        (spec.global_dims[0], "innermost global shape"),
        (spec.box_dims[0], "innermost box"),
        (spec.coordinates[0], "innermost coordinate"),
    ):
        if _proof_equal(tvm.tirx.floormod(value, 2), 0, analyzer) != ProofStatus.PROVEN:
            return None
    for stride in spec.global_strides:
        if (
            _proof_equal(tvm.tirx.floormod(stride, spec.effective_bytes * 2), 0, analyzer)
            != ProofStatus.PROVEN
        ):
            return None

    global_dims = list(spec.global_dims)
    box_dims = list(spec.box_dims)
    coordinates = list(spec.coordinates)
    global_dims[0] = analyzer.simplify(tvm.tirx.floordiv(global_dims[0], 2))
    box_dims[0] = analyzer.simplify(tvm.tirx.floordiv(box_dims[0], 2))
    coordinates[0] = analyzer.simplify(tvm.tirx.floordiv(coordinates[0], 2))
    new_spec = replace(
        spec,
        descriptor_dtype=new_dtype,
        descriptor_bits=new_bits,
        effective_bytes=new_bits // 8,
        global_dims=tuple(global_dims),
        box_dims=tuple(box_dims),
        coordinates=tuple(coordinates),
    )
    transaction_bits = new_bits
    for box in box_dims:
        transaction_bits = analyzer.simplify(transaction_bits * box)
    issue_count = 1
    for axis in plan.issue_axes:
        issue_count = analyzer.simplify(issue_count * axis.extent)
    new_spec = replace(
        new_spec,
        transaction_bits=transaction_bits,
        payload_bits=analyzer.simplify(transaction_bits * issue_count),
    )
    if _proof_equal(new_spec.payload_bits, spec.payload_bits, analyzer) != ProofStatus.PROVEN:
        return None
    return replace(plan, spec=new_spec)


def _repair_auto_candidate(plan: TMAPlan):
    candidate = plan
    while True:
        candidate = _canonicalize_auto_plan(candidate)
        failures = _validation_failures(candidate.spec, auto=True)
        if not failures:
            return candidate, ()
        if any(finding.rule not in _REPAIRABLE_RULES for finding in failures):
            return None, failures

        promoted = _promote_auto_once(candidate)
        if promoted is None:
            return None, failures
        candidate = promoted


def _runtime_config(op_call, sctx, direction: str, *, explicit: bool):
    allowed = _EXPLICIT_CONFIG if explicit else _COMMON_CONFIG
    unknown = sorted(set(op_call.config) - allowed)
    if unknown:
        fail(
            f"dispatch={'tma_explicit' if explicit else 'tma_auto'} does not support "
            f"config key(s) {unknown}"
        )

    if not explicit and op_call.config.get("oob") is not None:
        fail('tma_auto does not support non-default oob; use dispatch="tma_explicit"')
    if direction != "g2s" and op_call.config.get("oob") is not None:
        fail("TensorMap oob is only valid for explicit global-to-shared copies")

    cta_group = op_call.config.get("cta_group", 1)
    if isinstance(cta_group, IntImm):
        cta_group = int(cta_group)
    if cta_group not in (1, 2):
        fail(f"cta_group must be 1 or 2, got {cta_group}")

    cta_mask = op_call.config.get("cta_mask", 0)
    if isinstance(cta_mask, IntImm):
        cta_mask_value = int(cta_mask)
        if not 0 <= cta_mask_value <= 0xFFFF:
            fail(f"cta_mask must fit uint16, got {cta_mask_value}")
    elif isinstance(cta_mask, int):
        if not 0 <= cta_mask <= 0xFFFF:
            fail(f"cta_mask must fit uint16, got {cta_mask}")
    elif not isinstance(cta_mask, tvm.tirx.Expr):
        fail("cta_mask must be an integer or TIR expression")
    mbar = op_call.config.get("mbar")
    mbarrier_addr = op_call.config.get("mbarrier_addr", False)
    if isinstance(mbarrier_addr, IntImm):
        mbarrier_addr = bool(int(mbarrier_addr))
    if not isinstance(mbarrier_addr, bool | tvm.tirx.Expr):
        fail("mbarrier_addr must be bool or Expr")
    if direction == "g2s":
        if mbar is None:
            fail("global-to-shared TMA requires mbar")
    else:
        if mbar is not None:
            fail("mbar is only valid for global-to-shared TMA")
        if mbarrier_addr not in (False, None):
            fail("mbarrier_addr is only valid for global-to-shared TMA")
        if not (
            (isinstance(cta_mask, int) and cta_mask == 0)
            or (isinstance(cta_mask, IntImm) and int(cta_mask) == 0)
        ):
            fail("cta_mask is only valid for global-to-shared TMA")

    use_tma_reduce = op_call.config.get("use_tma_reduce")
    if use_tma_reduce is not None:
        if direction != "s2g":
            fail("use_tma_reduce is only valid for shared-to-global TMA")
        if use_tma_reduce not in ("add", "min", "max", "inc", "dec", "and", "or", "xor"):
            fail(f"unsupported TMA reduce operation {use_tma_reduce!r}")

    cache_hint, cache_policy = _normalize_cache_hint(op_call.config.get("cache_hint", ""))
    return {
        "cta_group": cta_group,
        "cta_mask": cta_mask,
        "mbar": mbar,
        "mbarrier_addr": mbarrier_addr,
        # A dynamic form selects between two equivalent address
        # representations.  Normalize it once to the shared-address form so
        # there remains exactly one TMA instruction.
        "mbar_is_shared_addr": (
            direction == "g2s"
            and (
                cta_group == 2 or mbarrier_addr is True or isinstance(mbarrier_addr, tvm.tirx.Expr)
            )
        ),
        "use_tma_reduce": use_tma_reduce,
        "cache_hint": cache_hint,
        "cache_policy": cache_policy,
        "l2_promotion": _normalize_l2_promotion(op_call.config.get("tensormap_l2_promotion")),
        "oob_fill": _normalize_oob(op_call.config.get("oob")),
        "prefetch": bool(op_call.config.get("prefetch_tensormap", False)),
        "tma_dtype": op_call.config.get("tma_dtype"),
        "target_arch": sctx.target.arch,
    }


def _copy_direction(op_call):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_region, src_region = op_call.dst, op_call.src
    src_scope = src_region.buffer.scope()
    dst_scope = dst_region.buffer.scope()
    if src_scope == "global" and dst_scope.startswith("shared"):
        return "g2s", dst_region, src_region
    if src_scope.startswith("shared") and dst_scope == "global":
        return "s2g", src_region, dst_region
    fail(f"TMA requires global<->shared operands, got src={src_scope}, dst={dst_scope}")


def _build_auto_plan(op_call: TilePrimitiveCall, sctx: DispatchContext) -> TMAPlan:
    direction, shared_region, global_region = _copy_direction(op_call)
    s_buf = shared_region.buffer
    g_buf = global_region.buffer
    if str(s_buf.dtype) != str(g_buf.dtype):
        _auto_fail(
            "dtype",
            f"shared dtype={s_buf.dtype} and global dtype={g_buf.dtype} differ",
        )
    runtime = _runtime_config(op_call, sctx, direction, explicit=False)
    if "gather4" in op_call.config or "src_selector" in op_call.config:
        fail('gather4 and src_selector are only supported by dispatch="tma_explicit"')

    s_starts, s_extents = _copy_region_parts(shared_region.region)
    g_starts, g_extents = _copy_region_parts(global_region.region)
    try:
        swizzle = get_swizzle_mode_from_layout(s_buf.layout)
    except ValueError as error:
        _auto_fail("shared-swizzle", str(error))
    if swizzle is None:
        _auto_fail("shared-layout", f"cannot recognize shared swizzle in {s_buf.layout}")
    try:
        _, sliced_smem_with_offset = _slice_layout(s_buf, s_starts, s_extents, "shared")
    except ValueError as error:
        _auto_fail("shared-slice", str(error))
    smem_layout_offset = _layout_offset(sliced_smem_with_offset)
    sliced_smem = TileLayout.from_iters(
        sliced_smem_with_offset.shard,
        sliced_smem_with_offset.replica,
        {},
    ).canonicalize()
    if not isinstance(sliced_smem, TileLayout):
        _auto_fail(
            "shared-canonicalize",
            f"canonical sliced shared layout is not a TileLayout: {sliced_smem}",
        )
    _assert_plain_memory_layout(sliced_smem, "canonical sliced shared")
    canonical_smem_shape = tuple(iterator.extent for iterator in sliced_smem.shard)
    try:
        sliced_smem, _ = sliced_smem.group_many((canonical_smem_shape, tuple(g_extents)))
    except (TypeError, ValueError, tvm.error.TVMError) as error:
        _auto_fail(
            "group-shared",
            f"cannot refine canonical shared shape={canonical_smem_shape} "
            f"with copy shape={tuple(g_extents)}: {error}",
        )
    smem_order = _smem_iter_order(sliced_smem)
    gt_records = _build_auto_gt(g_buf, g_starts, g_extents, sliced_smem, sctx)
    descriptor = _dtype_contract(g_buf.dtype, runtime["tma_dtype"])

    best = None
    last_failures = ()
    for prefix in range(1, len(smem_order) + 1):
        raw = _auto_spec_for_prefix(
            op_call=op_call,
            sctx=sctx,
            direction=direction,
            s_buf=s_buf,
            g_buf=g_buf,
            s_starts=s_starts,
            smem_layout_offset=smem_layout_offset,
            sliced_smem=sliced_smem,
            smem_order=smem_order,
            gt_records=gt_records,
            prefix=prefix,
            swizzle=swizzle,
            descriptor=descriptor,
            runtime=runtime,
        )
        _validate_auto_shared_mapping(raw, sliced_smem, swizzle, sctx)
        repaired, failures = _repair_auto_candidate(raw)
        if repaired is None:
            last_failures = failures
            if best is not None:
                break
            continue
        best = repaired

    if best is None:
        if last_failures:
            _raise_validation("prefix-search", last_failures, auto=True)
        _auto_fail("prefix-search", "no legal shared-memory prefix")
    return best


def _explicit_smem_layout(s_buf: Buffer, starts, extents, swizzle: SwizzleMode):
    _, sliced = _slice_layout(s_buf, starts, extents, "shared")
    # LayoutSlice preserves the layout's physical base and adds the selected
    # region offset, so the sliced offset is already the complete layout-side
    # contribution to the shared pointer.
    offset = _layout_offset(sliced)
    normalized = TileLayout.from_iters(sliced.shard, sliced.replica, {})
    canonical = normalized.canonicalize()
    if not canonical.is_trivial():
        fail(
            "tma_explicit stage=shared-layout: sliced shared layout must canonicalize "
            f"to trivial after extracting its pointer offset; got {canonical}"
        )
    analyzer = Analyzer()
    byte_offset = _elements_to_bytes(
        s_buf.elem_offset + offset,
        tvm.DataType(s_buf.dtype).bits,
        analyzer,
        auto=False,
        stage="shared-layout",
        label="shared pointer offset",
    )
    atom_alignment = {0: 16, 1: 32, 2: 64, 3: 128}[swizzle.value]
    status = _proof_equal(tvm.tirx.floormod(byte_offset, atom_alignment), 0, analyzer)
    if status == ProofStatus.DISPROVEN:
        fail(
            "tma_explicit stage=shared-layout: shared slice pointer offset "
            f"{byte_offset}B violates {atom_alignment}B swizzle/base alignment"
        )
    return (
        _to_tile_layout(s_buf.layout, s_buf.shape),
        analyzer.simplify(s_buf.elem_offset + offset),
    )


def _direct_global_layout(g_buf: Buffer):
    layout = g_buf.layout
    if not isinstance(layout, TileLayout):
        fail(
            "tma_explicit stage=global-layout: global Buffer/view layout must be "
            f"TileLayout, got {type(layout).__name__}"
        )
    _assert_plain_memory_layout(layout, "explicit global")
    if len(g_buf.shape) != len(layout.shard):
        fail(
            "tma_explicit stage=global-layout: tensor rank "
            f"{len(g_buf.shape)} != memory-layout rank {len(layout.shard)}"
        )
    analyzer = Analyzer()
    for dim, (shape, iterator) in enumerate(zip(g_buf.shape, layout.shard)):
        status = _proof_equal(shape, iterator.extent, analyzer)
        if status != ProofStatus.PROVEN:
            fail(
                "tma_explicit stage=global-layout: "
                f"dim={dim} shape={shape} must provably equal layout extent="
                f"{iterator.extent}; got {status.value}"
            )
    return layout


def _explicit_spec_for_gmem(
    *,
    g_buf,
    s_buf,
    s_starts,
    g_starts,
    g_extents,
    swizzle,
    runtime,
    sctx,
    gather4,
    smem_base_offset,
):
    analyzer = Analyzer()
    layout = _direct_global_layout(g_buf)
    descriptor = _dtype_contract(g_buf.dtype, runtime["tma_dtype"])
    descriptor_dtype, descriptor_bits, effective_bytes, packed_kind, force_cu_dtype = descriptor
    strides_outer = [
        _elements_to_bytes(
            iterator.stride,
            descriptor_bits,
            analyzer,
            auto=False,
            stage="global-layout",
            label=f"global layout stride dim={idx}",
        )
        for idx, iterator in enumerate(layout.shard)
    ]
    base, base_key, base_byte_offset = _buffer_base(g_buf, auto=False, stage="global-base")
    global_dims = tuple(reversed(g_buf.shape))
    box_dims = tuple(reversed(g_extents))
    coordinates = tuple(reversed(g_starts))
    full_strides = tuple(reversed(strides_outer))
    inner_stride = layout.shard[-1].stride
    if _proof_equal(inner_stride, 1, analyzer) != ProofStatus.PROVEN:
        fail(
            "tma_explicit stage=global-layout: the innermost memory stride must be "
            f"provably one because CUDA omits it from globalStrides; got {inner_stride}"
        )
    transaction_bits = descriptor_bits
    for box in box_dims:
        transaction_bits = analyzer.simplify(transaction_bits * box)
    if gather4:
        transaction_bits = analyzer.simplify(transaction_bits * 4)
    return TensorMapSpec(
        descriptor_dtype=descriptor_dtype,
        descriptor_bits=descriptor_bits,
        effective_bytes=effective_bytes,
        packed_kind=packed_kind,
        force_cu_dtype=force_cu_dtype,
        base=base,
        base_key=base_key,
        descriptor_name=g_buf.name,
        base_byte_offset=base_byte_offset,
        global_dims=global_dims,
        global_strides=full_strides[1:],
        inner_stride=inner_stride,
        box_dims=box_dims,
        element_strides=(1,) * len(global_dims),
        interleave=0,
        swizzle=swizzle.value,
        l2_promotion=runtime["l2_promotion"],
        oob_fill=runtime["oob_fill"],
        direction="g2s" if runtime["mbar"] is not None else "s2g",
        load_mode="tile_gather4" if gather4 else "tile",
        target_arch=sctx.target.arch,
        coordinates=coordinates,
        gather4=tuple(gather4),
        smem_buffer=s_buf,
        smem_start=tuple(s_starts),
        smem_base_offset=smem_base_offset,
        mbar=runtime["mbar"],
        mbar_is_shared_addr=runtime["mbar_is_shared_addr"],
        cta_group=runtime["cta_group"],
        cta_mask=runtime["cta_mask"],
        cache_hint=runtime["cache_hint"],
        cache_policy=runtime["cache_policy"],
        use_tma_reduce=runtime["use_tma_reduce"],
        payload_bits=transaction_bits,
        transaction_bits=transaction_bits,
    )


def _normalize_gather4(value):
    if value is None:
        return ()
    if not isinstance(value, list | tuple | tvm.ir.Array) or len(value) != 4:
        fail("tma_explicit gather4 must contain exactly four row coordinates")
    return tuple(value)


def _validate_gather4_dst(s_buf: Buffer, s_starts, s_extents, spec: TensorMapSpec) -> None:
    analyzer = Analyzer()
    if len(s_extents) < 1:
        fail("tma_explicit gather4 requires a non-scalar shared destination")
    if _proof_equal(s_extents[0], 4, analyzer) != ProofStatus.PROVEN:
        fail("tma_explicit gather4 requires public shared axis 0 to provably contain four rows")
    _, sliced = _slice_layout(s_buf, s_starts, s_extents, "gather4 shared")
    normalized = TileLayout.from_iters(sliced.shard, sliced.replica, {})
    canonical = normalized.canonicalize()
    if not canonical.is_trivial():
        fail(f"tma_explicit gather4 destination is not four-row box-linear: {canonical}")
    row_bits = spec.transaction_bits
    if spec.gather4:
        row_bits = analyzer.simplify(tvm.tirx.floordiv(row_bits, 4))
    shared_bits = tvm.DataType(s_buf.dtype).bits
    row_elements = analyzer.simplify(tvm.tirx.floordiv(row_bits, shared_bits))
    grouped, sep = sliced.group(list(s_extents))
    row_iters = grouped.shard[sep[0] : sep[1]]
    if len(row_iters) != 1:
        fail("tma_explicit gather4 destination row axis must map to one memory iterator")
    row = row_iters[0]
    if (
        _proof_equal(row.extent, 4, analyzer) != ProofStatus.PROVEN
        or _proof_equal(row.stride, row_elements, analyzer) != ProofStatus.PROVEN
    ):
        fail(
            "tma_explicit gather4 destination rows are not box-linear at "
            f"payload width {row_elements}; got extent={row.extent}, stride={row.stride}"
        )


def _normalize_src_selector(value):
    if value is None:
        return ()
    if not isinstance(value, list | tuple | tvm.ir.Array):
        fail("tma_explicit src_selector must be a list of (condition, global Buffer/view)")
    result = []
    for idx, item in enumerate(value):
        if not isinstance(item, list | tuple | tvm.ir.Array) or len(item) != 2:
            fail(f"tma_explicit src_selector[{idx}] must be a (condition, Buffer/view) pair")
        condition, buffer = item
        if not isinstance(condition, tvm.tirx.Expr):
            fail(f"tma_explicit src_selector[{idx}] condition must be a TIR expression")
        if not is_buffer_var(buffer):
            fail(
                f"tma_explicit src_selector[{idx}] candidate must be a global "
                "Buffer/view, not a region"
            )
        if buffer.scope() != "global":
            fail(
                f"tma_explicit src_selector[{idx}] candidate scope must be global, "
                f"got {buffer.scope()}"
            )
        result.append((condition, buffer))
    return tuple(result)


def _selector_compatibility(main: TensorMapSpec, candidate: TensorMapSpec, index: int):
    analyzer = Analyzer()
    fields = (
        ("descriptor dtype", main.descriptor_dtype, candidate.descriptor_dtype),
        ("descriptor bits", main.descriptor_bits, candidate.descriptor_bits),
        ("forced CUDA dtype", main.force_cu_dtype, candidate.force_cu_dtype),
        ("rank", main.rank, candidate.rank),
        ("box", main.box_dims, candidate.box_dims),
        ("interleave", main.interleave, candidate.interleave),
        ("swizzle", main.swizzle, candidate.swizzle),
        ("load mode", main.load_mode, candidate.load_mode),
        ("transaction bits", main.transaction_bits, candidate.transaction_bits),
    )
    for label, lhs, rhs in fields:
        if isinstance(lhs, tuple):
            equal = len(lhs) == len(rhs) and all(
                _proof_equal(a, b, analyzer) == ProofStatus.PROVEN for a, b in zip(lhs, rhs)
            )
        elif isinstance(lhs, int | str):
            equal = lhs == rhs
        else:
            equal = _proof_equal(lhs, rhs, analyzer) == ProofStatus.PROVEN
        if not equal:
            fail(
                f"tma_explicit src_selector[{index}] has incompatible {label}: "
                f"main={lhs}, candidate={rhs}"
            )


def _build_explicit_plan(op_call: TilePrimitiveCall, sctx: DispatchContext):
    direction, shared_region, global_region = _copy_direction(op_call)
    s_buf = shared_region.buffer
    g_buf = global_region.buffer
    runtime = _runtime_config(op_call, sctx, direction, explicit=True)
    gather4 = _normalize_gather4(op_call.config.get("gather4"))
    selectors = _normalize_src_selector(op_call.config.get("src_selector"))
    if (gather4 or selectors) and direction != "g2s":
        fail("tma_explicit gather4 and src_selector are only valid for global-to-shared")
    if gather4 and len(g_buf.shape) != 2:
        fail("tma_explicit gather4 requires a rank-2 global tensor")

    s_starts, s_extents = _copy_region_parts(shared_region.region)
    g_starts, g_extents = _copy_region_parts(global_region.region)
    if gather4:
        analyzer = Analyzer()
        if _proof_equal(g_extents[0], 1, analyzer) == ProofStatus.DISPROVEN:
            fail("tma_explicit gather4 global public axis 0 must describe one row")
        if _proof_equal(s_extents[0], 4, analyzer) == ProofStatus.DISPROVEN:
            fail("tma_explicit gather4 shared public axis 0 must describe four rows")

    try:
        swizzle = get_swizzle_mode_from_layout(s_buf.layout)
    except ValueError as error:
        fail(f"tma_explicit stage=shared-swizzle: {error}")
    if swizzle is None:
        fail(f"tma_explicit cannot recognize shared swizzle in {s_buf.layout}")
    shared_layout, smem_base_offset = _explicit_smem_layout(s_buf, s_starts, s_extents, swizzle)
    main_spec = _explicit_spec_for_gmem(
        g_buf=g_buf,
        s_buf=s_buf,
        s_starts=s_starts,
        g_starts=g_starts,
        g_extents=g_extents,
        swizzle=swizzle,
        runtime=runtime,
        sctx=sctx,
        gather4=gather4,
        smem_base_offset=smem_base_offset,
    )
    _validate_explicit(main_spec, "main-descriptor")
    if gather4:
        _validate_gather4_dst(s_buf, s_starts, s_extents, main_spec)

    candidate_specs = []
    for idx, (condition, candidate_buffer) in enumerate(selectors):
        candidate = _explicit_spec_for_gmem(
            g_buf=candidate_buffer,
            s_buf=s_buf,
            s_starts=s_starts,
            g_starts=g_starts,
            g_extents=g_extents,
            swizzle=swizzle,
            runtime=runtime,
            sctx=sctx,
            gather4=gather4,
            smem_base_offset=smem_base_offset,
        )
        _validate_explicit(candidate, f"src-selector[{idx}]")
        _selector_compatibility(main_spec, candidate, idx)
        candidate_specs.append((condition, candidate))
    return (
        TMAPlan(spec=main_spec, issue_axes=(), shared_layout=shared_layout),
        tuple(candidate_specs),
    )


def _descriptor_cache_key(spec: TensorMapSpec) -> str:
    fields = (
        spec.base_key,
        spec.descriptor_dtype,
        spec.descriptor_bits,
        spec.force_cu_dtype,
        spec.global_dims,
        spec.global_strides,
        spec.box_dims,
        spec.element_strides,
        spec.interleave,
        spec.swizzle,
        spec.l2_promotion,
        spec.oob_fill,
    )
    return "tensormap:" + ":".join(str(field) for field in fields)


def _get_or_encode_descriptor(spec: TensorMapSpec, sctx: DispatchContext):
    key = _descriptor_cache_key(spec)
    cached = sctx.cache_get(key)
    if cached is not None:
        return cached, key

    tensor_map = T.Var(f"{spec.descriptor_name}_tensormap", ty=T.handle("tensormap").ty)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def create_tensor_map():
        T.Bind(T.tvm_stack_alloca("tensormap", 1), var=tensor_map)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            tensor_map,
            spec.descriptor_dtype,
            spec.rank,
            spec.base,
            *spec.global_dims,
            *spec.global_strides,
            *spec.box_dims,
            *spec.element_strides,
            spec.interleave,
            spec.swizzle,
            spec.l2_promotion,
            spec.oob_fill,
            *([spec.force_cu_dtype] if spec.force_cu_dtype >= 0 else []),
        )
        T.tvm_kernel_replace_point()
    # fmt: on

    sctx.add_init_stmt(create_tensor_map.body, host=True)
    sctx.cache_set(key, tensor_map)
    return tensor_map, key


def _prefetch_main_descriptor(tensor_map, key: str, sctx: DispatchContext) -> None:
    cache_key = f"prefetch_tensormap:{key}"
    if sctx.cache_get(cache_key) is not None:
        return
    if "warp_id_in_cta" not in sctx.launch_params:
        fail("prefetch_tensormap requires warp_id_in_cta launch param")
    warp_id = sctx.launch_params["warp_id_in_cta"].var

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def prefetch_tensor_map():
        if warp_id == 0:
            if T.ptx.elect_sync() != T.uint32(0):
                T.ptx.prefetch_tensormap(T.address_of(tensor_map))
        T.tvm_kernel_replace_point()
    # fmt: on

    sctx.add_init_stmt(prefetch_tensor_map.body)
    sctx.cache_set(cache_key, tensor_map)


def _selected_descriptor(main_map, candidate_maps):
    if not candidate_maps:
        return main_map, None, False
    selected_expr = T.address_of(main_map)
    for condition, candidate_map in reversed(candidate_maps):
        selected_expr = tvm.tirx.Select(
            condition,
            T.address_of(candidate_map),
            selected_expr,
        )
    selected = T.Var("selected_tensormap", "uint64")
    return selected, tvm.tirx.Bind(selected, selected_expr), True


def _emit_plan(
    plan: TMAPlan,
    tensor_map,
    selector_bind,
    tensor_map_is_address: bool,
    sctx: DispatchContext,
) -> PrimFunc:
    spec = plan.spec
    tensor_map_address = tensor_map if tensor_map_is_address else T.address_of(tensor_map)

    def tma_coordinates(coordinates):
        if spec.load_mode == "tile_gather4":
            return [coordinates[0], *spec.gather4]
        return list(coordinates)

    def emit_at(shared_ptr, coordinates):
        coords = tma_coordinates(coordinates)
        if spec.direction == "g2s":
            mbar_operand = (
                T.cuda.cvta_generic_to_shared(spec.mbar) if spec.mbar_is_shared_addr else spec.mbar
            )
            T.evaluate(
                T.ptx.cp_async.bulk.tensor.g2s_cluster(
                    spec.rank,
                    shared_ptr,
                    mbar_operand,
                    tensor_map_address,
                    spec.cta_mask,
                    spec.cta_group,
                    spec.cache_hint,
                    *coords,
                    cache_policy=spec.cache_policy,
                    load_mode=spec.load_mode,
                    mbar_is_shared_addr=spec.mbar_is_shared_addr,
                )
            )
        elif spec.use_tma_reduce is None:
            T.evaluate(
                T.ptx.cp_async.bulk.tensor.s2g(
                    spec.rank,
                    shared_ptr,
                    tensor_map_address,
                    spec.cache_hint,
                    *coords,
                    cache_policy=spec.cache_policy,
                )
            )
        else:
            T.evaluate(
                T.ptx.cp_async.bulk.tensor.s2g_reduce(
                    spec.rank,
                    shared_ptr,
                    tensor_map_address,
                    spec.cache_hint,
                    spec.use_tma_reduce,
                    *coords,
                    cache_policy=spec.cache_policy,
                )
            )

    def shared_ptr(element_offset=0):
        # Keep the sliced offset in the pointer index instead of the Buffer's
        # elem_offset.  The latter is part of a flat DeclBuffer definition;
        # after loop unrolling, CSE may otherwise lift an offset containing a
        # locally bound coordinate above that coordinate's Bind statement.
        smem_view = T.decl_buffer(
            (1,),
            spec.smem_buffer.dtype,
            spec.smem_buffer.data,
            elem_offset=0,
            scope=spec.smem_buffer.scope(),
        )
        return smem_view.ptr_to([spec.smem_base_offset + element_offset])

    if not plan.issue_axes:
        # fmt: off
        @T.prim_func(check_well_formed=False)
        def impl():
            emit_at(shared_ptr(), spec.coordinates)
        # fmt: on

    else:
        flat_extent = plan.issue_extent()

        # fmt: off
        @T.prim_func(check_well_formed=False)
        def impl():
            for issue in T.unroll(flat_extent):
                smem_offset, coordinates = T.meta_var(plan.offsets_and_coords(issue))
                simplified = T.meta_var(
                    _simplify_with_ranges(
                        [smem_offset, *coordinates], [(issue, flat_extent)], sctx
                    )
                )
                emit_at(
                    shared_ptr(simplified[0]),
                    simplified[1:],
                )
        # fmt: on

    if selector_bind is not None:
        body = tvm.tirx.SeqStmt([selector_bind, impl.body])
        impl = PrimFunc([], body, ret_type=None, buffer_map={}).with_attr("global_symbol", "impl")
    return impl


def copy_tma_auto_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower one ``tma_auto`` call."""

    plan = _build_auto_plan(op_call, sctx)
    tensor_map, key = _get_or_encode_descriptor(plan.spec, sctx)
    impl = _emit_plan(plan, tensor_map, None, False, sctx)
    if bool(op_call.config.get("prefetch_tensormap", False)):
        _prefetch_main_descriptor(tensor_map, key, sctx)
    return impl


def copy_tma_explicit_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower one direct TensorMap and exactly one TMA instruction."""

    plan, candidates = _build_explicit_plan(op_call, sctx)
    main_map, main_key = _get_or_encode_descriptor(plan.spec, sctx)
    candidate_maps = []
    for condition, candidate_spec in candidates:
        candidate_map, _ = _get_or_encode_descriptor(candidate_spec, sctx)
        candidate_maps.append((condition, candidate_map))
    selected_map, selector_bind, selected_is_address = _selected_descriptor(
        main_map, candidate_maps
    )
    impl = _emit_plan(plan, selected_map, selector_bind, selected_is_address, sctx)
    if bool(op_call.config.get("prefetch_tensormap", False)):
        _prefetch_main_descriptor(main_map, main_key, sctx)
    return impl


def _validate_tma_copy_op(op_call: TilePrimitiveCall, _sctx: DispatchContext) -> bool:
    dst_region, src_region = op_call.args[:2]
    src = src_region.buffer
    dst = dst_region.buffer
    if src.layout is None or dst.layout is None:
        return False
    src_scope, dst_scope = src.scope(), dst.scope()
    return (src_scope == "global" and dst_scope.startswith("shared")) or (
        src_scope.startswith("shared") and dst_scope == "global"
    )


_COMMON_PREDICATES = [
    predicate(
        "validate_tma_copy_op",
        lambda op, sctx: (_validate_tma_copy_op(op, sctx), "not a global<->shared TMA copy"),
    ),
    predicate(
        "single_thread",
        lambda op, sctx: (
            single_thread(op, sctx),
            f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
        ),
    ),
]


@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma_auto",
    priority=10,
    when=_COMMON_PREDICATES,
)
def copy_async_dispatch_tma_auto(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tma_auto_impl(op, sctx)


@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma_explicit",
    priority=10,
    when=_COMMON_PREDICATES,
)
def copy_async_dispatch_tma_explicit(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tma_explicit_impl(op, sctx)
