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

"""Instruction generation utilities for TRN operator scheduling."""

import itertools
from dataclasses import dataclass
from functools import reduce
from math import gcd
from operator import mul

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.ir import Range
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimExpr, Var
from tvm.tirx.expr_functor import ExprMutator
from tvm.tirx.layout import Iter

from .dim_utils import DimensionMapper, RangeInfo, normalize_and_group


@dataclass
class LogicalIterDim:
    logical_stride: int
    extent: int
    bind_expr: PrimExpr

    @staticmethod
    def default():
        return LogicalIterDim(1, 1, T.int32(0))


LogicalIterList = tuple[tuple[tuple[LogicalIterDim]]]


def to_int_list(intimm_list: list[T.IntImm]):
    return [int(i) for i in intimm_list]


class VarReplacer(ExprMutator):
    def __init__(self, var_map: dict[Var, PrimExpr]):
        super().__init__()
        self.var_map = var_map

    def visit_var_(self, op):
        if op in self.var_map:
            return self.var_map[op]
        return op

    @staticmethod
    def replace_vars(expr: PrimExpr, var_map: dict[Var, PrimExpr]) -> PrimExpr:
        return VarReplacer(var_map).visit_expr(expr)


@dataclass
class InstructionRepr:
    buffer_region: BufferRegion
    size: int
    stride: int
    selected_data_iter_ids: list[int]

    def __init__(
        self,
        buffer_region: BufferRegion,
        inst_size: int,
        inst_stride: int,
        selected_data_iter_ids: list[int],
    ):
        self.buffer_region = buffer_region
        self.size = inst_size if inst_size is not None else 1
        self.stride = inst_stride if inst_stride is not None else 1
        self.selected_data_iter_ids = selected_data_iter_ids

    def bound_inst_size(self, max_inst_size: int | None, analyzer: Analyzer):
        if max_inst_size is None:
            return
        if analyzer.can_prove(self.size <= max_inst_size):
            return
        assert analyzer.can_prove(self.size % max_inst_size == 0), (
            f"The instruction size {self.size} is not a multiple of the max instruction size {max_inst_size}"  # noqa: E501
        )
        self.size = max_inst_size
        self.selected_data_iter_ids = None


class InstructionGenerator:
    def __init__(self, buffer_regions: tuple[BufferRegion], analyzer: Analyzer):
        self.buffer_regions = []
        self.analyzer = analyzer
        self.split_shape_views = {}
        self.split_layout_views = {}
        self.seps = {}
        self.bound_regions = {}
        self.bind_iters: dict[BufferRegion, LogicalIterList] = None
        self.bind_maps: dict[BufferRegion, dict[Var, PrimExpr]] = {}
        for buffer_region in buffer_regions:
            if not isinstance(buffer_region, BufferRegion):
                continue
            self.buffer_regions.append(buffer_region)
            bound_buffer_region = self._bound_buffer_region(buffer_region)
            layout, seps = self._get_sub_layout(bound_buffer_region)
            self.split_shape_views[buffer_region] = self._get_flattened_shape_view_from_layout_seps(
                layout, seps
            )
            self.split_layout_views[buffer_region] = layout
            self.seps[buffer_region] = seps
        self.dim_mapper = DimensionMapper()

    def _bound_buffer_region(self, buffer_region: BufferRegion):
        region = []
        changed = False
        for r in buffer_region.region:
            bound = self.analyzer.const_int_bound(r.extent)
            if not self.analyzer.can_prove_equal(bound.max_value, r.extent):
                changed = True
            region.append(Range.from_min_extent(r.min, bound.max_value))
        if changed:
            bound_region = BufferRegion(buffer_region.buffer, region)
            self.bound_regions[buffer_region] = bound_region
            return bound_region
        return buffer_region

    def _get_sub_layout(self, buffer_region: BufferRegion):
        layout = buffer_region.buffer.layout
        layout, seps = normalize_and_group(layout, buffer_region.buffer.shape)
        tiled_range_infos_per_dim = []
        new_shard = []
        new_seps = [0]
        for i in range(len(seps) - 1):
            r = buffer_region.region[i]
            st = r.min
            ext = r.extent
            reversed_shard = []
            for j in reversed(range(seps[i], seps[i + 1])):
                if self.analyzer.can_prove_equal(ext, 1):
                    break
                if layout.shard[j].axis.name == "P" and (
                    not self.analyzer.can_prove(st % layout.shard[j].extent == 0)
                    or not self.analyzer.can_prove(ext % layout.shard[j].extent == 0)
                ):
                    assert False, "Invalid layout"
                if self.analyzer.can_prove(
                    ext % layout.shard[j].extent == 0
                ) and self.analyzer.can_prove(st % layout.shard[j].extent == 0):
                    st = st // layout.shard[j].extent
                    ext = ext // layout.shard[j].extent
                    tiled_range_infos_per_dim.append(
                        RangeInfo(0, layout.shard[j].extent, j, i, layout.shard[j].axis)
                    )
                    reversed_shard.append(layout.shard[j])
                    continue
                if self.analyzer.can_prove(st + ext <= layout.shard[j].extent):
                    tiled_range_infos_per_dim.append(RangeInfo(st, ext, j, i, layout.shard[j].axis))
                    reversed_shard.append(Iter(ext, layout.shard[j].stride, layout.shard[j].axis))
                    break
                assert False, f"Cannot analyze physical tensor region for: {buffer_region}"
            new_shard += reversed(reversed_shard)
            new_seps.append(len(reversed_shard) + new_seps[-1])
        new_tile_layout = tvm.tirx.layout.TileLayout.from_iters(  # pylint: disable=no-member
            new_shard, [], dict()
        )
        return new_tile_layout, new_seps

    def _init_bind_iters(self):
        self.bind_iters = {}
        for buffer_region in self.buffer_regions:
            seps = self.seps[buffer_region]
            self.bind_iters[buffer_region] = [
                [[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))
            ]

    def _normalize_bind_iters(self):
        for buffer_region in self.buffer_regions:
            seps = self.seps[buffer_region]
            self.bind_iters[buffer_region] = [
                [
                    sorted(
                        self.bind_iters[buffer_region][i][j - seps[i]],
                        key=lambda x: (x.logical_stride, x.extent),
                    )
                    for j in range(seps[i], seps[i + 1])
                ]
                for i in range(len(buffer_region.region))
            ]

    def _get_flattened_shape_view_from_layout_seps(self, layout, seps):
        return [
            [layout.shard[j].extent for j in range(seps[i], seps[i + 1])]
            for i in range(len(seps) - 1)
        ]

    def common_factor(self, shape_a, shape_b):
        """
        Return the finest common factor shape of two compatible shapes.

        A "common factor" shape `C` satisfies:
        1. ∏shape_a == ∏shape_b == ∏C   (same total #elements)
        2. `C` can be obtained from `shape_a` **only** by splitting (never merging)
            dimensions, and likewise for `shape_b`.

        Parameters
        ----------
        shape_a, shape_b : tuple[int] | list[int]
            Two equally-sized shapes.

        Returns
        -------
        tuple[int]
            The common-factor shape.

        Raises
        ------
        AssertionError
            - if the shapes have different element counts
            - or if a common-factor decomposition does not exist
            (which only happens if the two shapes do not share a
            compatible prime-factor ordering).
        """
        if len(shape_a) == 0 and len(shape_b) == 0:
            return shape_a
        shape_a = to_int_list(shape_a)
        shape_b = to_int_list(shape_b)
        # 1. identical element count
        size_a = reduce(mul, shape_a, 1)
        size_b = reduce(mul, shape_b, 1)
        assert size_a == size_b, "Shapes hold different numbers of elements"

        i, j = 0, 0
        rem_a, rem_b = shape_a[0], shape_b[0]
        out = []

        while i < len(shape_a) and j < len(shape_b):
            g = gcd(rem_a, rem_b)
            assert g > 1 or (rem_a == rem_b == 1), "Incompatible factor ordering"
            out.append(g)

            # consume g from the current "head" factors
            rem_a //= g
            rem_b //= g

            # advance whenever a remainder has been completely consumed
            if rem_a == 1:
                i += 1
                rem_a = shape_a[i] if i < len(shape_a) else 1
            if rem_b == 1:
                j += 1
                rem_b = shape_b[j] if j < len(shape_b) else 1

        # sanity check
        assert i == len(shape_a) and j == len(shape_b), "Did not exhaust both shapes"

        return tuple(out)

    def _link_buffer_regions(
        self, buffer_region: BufferRegion, to_link: BufferRegion, dim_map: dict[int, int]
    ):
        split_shape_view_1 = self.split_shape_views[buffer_region]
        split_layout_view_1 = self.split_layout_views[buffer_region]
        split_shape_view_2 = self.split_shape_views[to_link]

        # adapt to the shape view of the to_link buffer region
        new_split_shape_view_1 = [
            (
                self.common_factor(split_shape_view_2[dim_map[i]], split_shape_view_1[i])
                if i in dim_map
                else split_shape_view_1[i]
            )
            for i in range(len(buffer_region.region))
        ]
        flattened_shape_view_1 = list(itertools.chain(*new_split_shape_view_1))
        layout, tiled_seps = normalize_and_group(split_layout_view_1, flattened_shape_view_1)
        actual_seps = [0]
        ptr = 0
        for i in range(len(buffer_region.region)):
            ptr += len(new_split_shape_view_1[i])
            actual_seps.append(tiled_seps[ptr])
        self.split_shape_views[buffer_region] = self._get_flattened_shape_view_from_layout_seps(
            layout, actual_seps
        )
        self.split_layout_views[buffer_region] = layout
        self.seps[buffer_region] = actual_seps

    def _get_reverse_dim_map(self, dim_map: dict[int, int]) -> dict[int, int]:
        return {dim_map[i]: i for i in dim_map}

    def link_buffer_regions(
        self, buffer_region: BufferRegion, to_link: BufferRegion, dim_map: dict[int, int]
    ):
        self.dim_mapper.register_dim_map(buffer_region, to_link, dim_map)
        for r in self.buffer_regions:
            if r == to_link:
                continue
            dim_map = self.dim_mapper.get_dim_map(r, to_link)
            reverse_dim_map = self._get_reverse_dim_map(dim_map)
            self._link_buffer_regions(r, to_link, dim_map)
            self._link_buffer_regions(to_link, r, reverse_dim_map)
            seps_1 = self.seps[r]
            seps_2 = self.seps[to_link]
            for i, j in dim_map.items():
                assert seps_1[i + 1] - seps_1[i] == seps_2[j + 1] - seps_2[j], (
                    f"The number of data iters at dim {i} of {buffer_region.buffer.name} is not equal to the number of data iters at dim {j} of {to_link.buffer.name}"  # noqa: E501
                )

    def bind_inst_iter(
        self,
        buffer_region: BufferRegion,
        bind: Var,
        inst_size: int,
        inst_stride: int,
        is_free_dim: bool,
        no_propagate: bool = False,
    ):
        logical_iter_list = self._get_inst_logical_iter_list(
            buffer_region, bind, inst_stride, inst_size, is_free_dim
        )
        self._add_bind_iter_list(buffer_region, logical_iter_list)
        if no_propagate:
            return
        self._propagate_bind_iter(buffer_region, logical_iter_list)

    def _propagate_bind_iter(self, buffer_region: BufferRegion, logical_iter_list: LogicalIterList):
        for to_propagate in self.buffer_regions:
            if to_propagate == buffer_region:
                continue
            dim_map = self.dim_mapper.get_dim_map(buffer_region, to_propagate)
            reverse_dim_map = self._get_reverse_dim_map(dim_map)
            seps = self.seps[to_propagate]
            propagated_logical_iter = [
                (
                    logical_iter_list[reverse_dim_map[i]]
                    if i in reverse_dim_map
                    else [[] for _ in range(seps[i], seps[i + 1])]
                )
                for i in range(len(to_propagate.region))
            ]
            self._add_bind_iter_list(to_propagate, propagated_logical_iter)

    def _add_bind_iter_list(self, buffer_region: BufferRegion, bind_iter_list: LogicalIterList):
        if self.bind_iters is None:
            self._init_bind_iters()
        seps = self.seps[buffer_region]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                self.bind_iters[buffer_region][i][j - seps[i]].extend(
                    bind_iter_list[i][j - seps[i]]
                )

    def fill_in_block_dim(
        self, buffer_region: BufferRegion, bind: Var, dims: list[int] | None = None
    ):
        # fixme: be cautious of the min of buffer region. This implementation is not correct.
        #        we need to first take a view of sub-layout (keep strides, but reduce the extent
        #        then we analyze the relationship between data iter of sub-layout
        dims = dims or list(range(len(buffer_region.buffer.shape)))
        layout = self.split_layout_views[buffer_region]
        shards = layout.shard
        self._normalize_bind_iters()
        bind_iters = self.bind_iters[buffer_region]
        seps = self.seps[buffer_region]
        logical_iter_list_block = [
            [[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))
        ]
        acc_block_ext = 1
        for i in reversed(dims):
            for j in reversed(range(seps[i], seps[i + 1])):
                it = shards[j]
                is_partition = it.axis.name == "P" if layout.is_trainium() else False
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in range(-1, len(logical_iter_dims)):
                    next_logical_stride = (
                        logical_iter_dims[d + 1].logical_stride
                        if d + 1 < len(logical_iter_dims)
                        else it.extent
                    )
                    cur = (
                        logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent
                        if d >= 0
                        else 1
                    )
                    assert next_logical_stride % cur == 0, (
                        f"Fail to infer block dim for {buffer_region.buffer.name} at dim {i}"
                    )
                    gap = next_logical_stride // cur
                    if is_partition:
                        assert gap == 1, (
                            f"Fail to propagate partition dim. The propagated dim does not cover the whole partition on {buffer_region.buffer.name} at dim {i}"  # noqa: E501
                        )
                    elif gap > 1:
                        new_acc_block_ext = acc_block_ext * gap
                        logical_iter_list_block[i][j - seps[i]].append(
                            LogicalIterDim(cur, gap, bind % new_acc_block_ext // acc_block_ext)
                        )
                        acc_block_ext = new_acc_block_ext
        self._add_bind_iter_list(buffer_region, logical_iter_list_block)
        self._propagate_bind_iter(buffer_region, logical_iter_list_block)
        return acc_block_ext

    def _check_bind_iter_coverage(self, buffer_region: BufferRegion):
        self._normalize_bind_iters()
        seps = self.seps[buffer_region]
        iters = self.split_layout_views[buffer_region].shard
        bind_iters = self.bind_iters[buffer_region]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                it = iters[j]
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in range(len(logical_iter_dims)):
                    next_logical_stride = (
                        logical_iter_dims[d + 1].logical_stride
                        if d + 1 < len(logical_iter_dims)
                        else it.extent
                    )
                    assert (
                        next_logical_stride
                        % (logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent)
                        == 0
                    ), f"Fail to infer block dim for {buffer_region.buffer.name} at dim {i}"
                    gap = next_logical_stride // (
                        logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent
                    )
                    assert gap == 1, "Call fill_in_block_dim() before calling generate_indices()"

    def set_bind_map(self, buffer_region: BufferRegion, bind_map: dict[Var, PrimExpr]):
        self.bind_maps[buffer_region] = bind_map

    def set_bind_map_all(self, bind_map: dict[Var, PrimExpr]):
        for buffer_region in self.buffer_regions:
            self.set_bind_map(buffer_region, bind_map)

    def generate_axes(self, buffer_region: BufferRegion) -> list[PrimExpr]:
        self._check_bind_iter_coverage(buffer_region)
        layout = self.split_layout_views[buffer_region]
        iters = layout.shard
        bind_iters = self.bind_iters[buffer_region]
        seps = self.seps[buffer_region]
        axes = []
        for i in range(len(bind_iters)):
            index = 0
            acc_logical_stride = 1
            for j in reversed(range(seps[i], seps[i + 1])):
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in reversed(logical_iter_dims):
                    if d.extent == 1:
                        continue
                    index += (
                        d.logical_stride
                        * VarReplacer.replace_vars(d.bind_expr, self.bind_maps[buffer_region])
                        * acc_logical_stride
                    )
                acc_logical_stride *= iters[j].extent
            axes.append(index)
        return axes

    def generate_indices(self, buffer_region: BufferRegion) -> list[PrimExpr]:
        axes = self.generate_axes(buffer_region)
        return [axes[i] + r.min for i, r in enumerate(buffer_region.region)]

    def _get_inst_logical_iter_list(
        self,
        buffer_region: BufferRegion,
        bind: Var,
        stride: int,
        size: int,
        is_free_dim: bool = True,
    ) -> LogicalIterList:
        layout = self.split_layout_views[buffer_region]
        assert layout.is_trainium(), " Cannot propagate instruction information from HBM tensor"
        iters = layout.shard
        seps = self.seps[buffer_region]
        ret = [[[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                if (iters[j].axis.name in ["F", "Bank"]) ^ is_free_dim:
                    continue
                it = iters[j]
                if it.stride * it.extent <= stride or it.stride >= size * stride:
                    continue
                if it.stride * it.extent < size * stride and stride <= it.stride:
                    assert (size * stride) % (
                        it.stride * it.extent
                    ) == 0 and it.stride % stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(
                            1,
                            it.extent,
                            bind % (it.stride * it.extent // stride) // (it.stride // stride),
                        )
                    )
                elif it.stride * it.extent < size * stride and stride > it.stride:
                    assert (size * stride) % (
                        it.stride * it.extent
                    ) == 0 and stride % it.stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(
                            stride // it.stride,
                            it.stride * it.extent // stride,
                            bind % (it.stride * it.extent // stride),
                        )
                    )
                elif it.stride * it.extent >= size * stride and stride <= it.stride:
                    assert (it.stride * it.extent) % (
                        size * stride
                    ) == 0 and it.stride % stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(1, size * stride // it.stride, bind // (it.stride // stride))
                    )
        return ret

    def make_guard(self, buffer_region: BufferRegion):
        if buffer_region not in self.bound_regions:
            return True
        bound_region = self.bound_regions[buffer_region]
        relaxed_dims = [
            i
            for i, (r1, r2) in enumerate(zip(bound_region.region, buffer_region.region))
            if not self.analyzer.can_prove(r1.extent == r2.extent)
        ]
        axes = self.generate_axes(buffer_region)
        guard = reduce(
            T.And,
            [axes[i] < r.extent for i, r in enumerate(buffer_region.region) if i in relaxed_dims],
            True,
        )
        return guard

    def _find_max_linear_inst(self, indexed_data_iters, min_stride: int | None = None):
        min_stride = min_stride or 1
        indexed_data_iters = sorted(indexed_data_iters, key=lambda x: x[1].stride)
        inst_size = 1
        inst_stride = None
        idx_list = []
        for idx, data_iter in indexed_data_iters:
            if data_iter.extent == 1 or data_iter.stride * data_iter.extent < min_stride:
                continue
            assert data_iter.stride % min_stride == 0 or min_stride % data_iter.stride == 0, (
                f"Invalid instruction stride {min_stride}"
            )
            if inst_stride is not None and inst_stride * inst_size != data_iter.stride:
                # the stride of the found data iter is not compatible with previous data iters
                break
            elif inst_stride is None:
                inst_stride = max(min_stride, data_iter.stride)
            if min_stride % data_iter.stride == 0:
                inst_size = data_iter.extent * data_iter.stride // inst_stride
            else:
                inst_size *= data_iter.extent
            idx_list.append(idx)
        return inst_size, inst_stride, idx_list

    def find_max_inst_size_from_one_region(
        self,
        buffer_region: BufferRegion,
        allowed_f_dim: tuple[int] | None = None,
        min_stride: int | None = None,
    ):
        allowed_f_dim = allowed_f_dim or tuple(range(len(buffer_region.region)))
        layout = self.split_layout_views[buffer_region]
        seps = self.seps[buffer_region]
        allowed_data_iter_idx = itertools.chain.from_iterable(
            range(seps[dim], seps[dim + 1]) for dim in allowed_f_dim
        )
        filtered_data_iters = [
            (i, layout.shard[i])
            for i in allowed_data_iter_idx
            if layout.shard[i].axis.name in ["F", "Bank"]
        ]
        inst_size, inst_stride, idx_list = self._find_max_linear_inst(
            filtered_data_iters, min_stride
        )
        return InstructionRepr(buffer_region, inst_size, inst_stride, idx_list)

    def fit_inst_tile_to_region(
        self,
        inst_repr: InstructionRepr,
        to_region: BufferRegion,
        allowed_to_f_dim: tuple[int] | None = None,
        broadcast: bool = False,
    ):
        allowed_to_f_dim = allowed_to_f_dim or tuple(range(len(to_region.region)))
        from_region = inst_repr.buffer_region
        from_layout = self.split_layout_views[from_region]
        to_layout = self.split_layout_views[to_region]
        from_seps = self.seps[from_region]
        to_seps = self.seps[to_region]
        dim_map = self.dim_mapper.get_dim_map(from_region, to_region)
        dim_map = {i: j for i, j in dim_map.items() if j in allowed_to_f_dim}
        data_iter_map = {
            from_seps[i] + idx: to_seps[j] + idx
            for i, j in dim_map.items()
            for idx in range(from_seps[i + 1] - from_seps[i])
        }
        if broadcast:
            data_iter_idx_to_dim = {
                from_seps[i] + j: i
                for i in range(len(from_region.region))
                for j in range(from_seps[i + 1] - from_seps[i])
            }
            indexed_selected_shard = [
                (i, from_layout.shard[i])
                for i in inst_repr.selected_data_iter_ids
                if data_iter_idx_to_dim[i] not in dim_map
            ]
            inst_size, inst_stride, idx_list = self._find_max_linear_inst(indexed_selected_shard)
            return InstructionRepr(from_region, inst_size, inst_stride, idx_list)
        indexed_selected_shard = [
            (i, from_layout.shard[i]) for i in inst_repr.selected_data_iter_ids
        ]
        indexed_selected_shard = sorted(indexed_selected_shard, key=lambda x: x[1].stride)
        inst_size = 1
        inst_stride_from = None
        inst_stride_to = None
        idx_list = []
        for i, data_iter in indexed_selected_shard:
            if i not in data_iter_map:
                if inst_stride_from is None:
                    continue
                break
            mapped_data_iter = to_layout.shard[data_iter_map[i]]
            if inst_stride_from is None:
                inst_stride_from = data_iter.stride
                if not to_layout.is_trainium() and mapped_data_iter.stride != 1:
                    # dma copy must be contiguous on hbm
                    break
                inst_stride_to = mapped_data_iter.stride
            elif inst_stride_to * inst_size != mapped_data_iter.stride:
                break
            inst_size *= data_iter.extent
            idx_list.append(i)
        return InstructionRepr(from_region, inst_size, inst_stride_from, idx_list)

    def check_partition_dim_match(
        self, buffer_region_1: BufferRegion, buffer_region_2: BufferRegion
    ):
        dim_map = self.dim_mapper.get_dim_map(buffer_region_1, buffer_region_2)
        layout_1 = self.split_layout_views[buffer_region_1]
        layout_2 = self.split_layout_views[buffer_region_2]
        if not layout_1.is_trainium() or not layout_2.is_trainium():
            return True
        seps_1 = self.seps[buffer_region_1]
        seps_2 = self.seps[buffer_region_2]
        for i, j in dim_map.items():
            for k in range(seps_1[i + 1] - seps_1[i]):
                if (
                    layout_1.shard[seps_1[i] + k].axis.name
                    != layout_2.shard[seps_2[j] + k].axis.name
                ):
                    return False
                if layout_1.shard[seps_1[i] + k].axis.name in ["F", "Bank"]:
                    continue
                if layout_1.shard[seps_1[i] + k].stride != layout_2.shard[seps_2[j] + k].stride:
                    return False
                if layout_1.shard[seps_1[i] + k].extent != layout_2.shard[seps_2[j] + k].extent:
                    return False
        return True

    def find_max_inst_size_transpose(
        self, buffer_region_1: BufferRegion, buffer_region_2: BufferRegion
    ):
        dim_map = self.dim_mapper.get_dim_map(buffer_region_1, buffer_region_2)
        layout_1 = self.split_layout_views[buffer_region_1]
        layout_2 = self.split_layout_views[buffer_region_2]
        iters_1 = layout_1.shard
        iters_2 = layout_2.shard
        seps_1 = self.seps[buffer_region_1]
        seps_2 = self.seps[buffer_region_2]
        indexed_iters_1 = []
        indexed_iters_2 = []
        print(iters_1, seps_1)
        print(iters_2, seps_2)
        print(dim_map)
        for i, j in dim_map.items():
            for k in range(seps_1[i + 1] - seps_1[i]):
                if iters_1[seps_1[i] + k].axis.name == iters_2[seps_2[j] + k].axis.name:
                    if iters_1[seps_1[i] + k].axis.name in ["F", "Bank"]:
                        continue
                    raise ValueError("Transpose only part of P dimension is not supported")
                if iters_1[seps_1[i] + k].axis.name == "P":
                    indexed_iters_2.append((seps_2[j] + k, iters_2[seps_2[j] + k]))
                else:
                    indexed_iters_1.append((seps_1[i] + k, iters_1[seps_1[i] + k]))
        inst_repr_1 = InstructionRepr(buffer_region_1, *self._find_max_linear_inst(indexed_iters_1))
        inst_repr_2 = InstructionRepr(buffer_region_2, *self._find_max_linear_inst(indexed_iters_2))
        assert inst_repr_1.size == layout_2.size("P"), (
            f"The instruction size of {buffer_region_1.buffer.name} does not match the partition size of {buffer_region_2.buffer.name}"  # noqa: E501
        )
        assert inst_repr_2.size == layout_1.size("P"), (
            f"The instruction size of {buffer_region_2.buffer.name} does not match the partition size of {buffer_region_1.buffer.name}"  # noqa: E501
        )
        return inst_repr_1, inst_repr_2

    def restrict_inst_to_one_dim(self, inst_repr: InstructionRepr):
        region = inst_repr.buffer_region
        layout = self.split_layout_views[region]
        iters = layout.shard
        seps = self.seps[region]
        indexed_selected_iters = [(i, iters[i]) for i in inst_repr.selected_data_iter_ids]
        indexed_selected_iters = sorted(indexed_selected_iters, key=lambda x: x[1].stride)
        iter_idx_to_dim = {
            seps[j]: i for i in range(len(region.buffer.shape)) for j in range(seps[i], seps[i + 1])
        }
        last_dim = None
        inst_size = 1
        selected_data_iter_ids = []
        for i, it in indexed_selected_iters:
            if last_dim is None:
                inst_size *= it.extent
                last_dim = iter_idx_to_dim[i]
                selected_data_iter_ids.append(i)
                continue
            if iter_idx_to_dim[i] != last_dim:
                break
            inst_size *= it.extent
            selected_data_iter_ids.append(i)
        return InstructionRepr(region, inst_size, inst_repr.stride, selected_data_iter_ids)
