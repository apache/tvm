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

"""Layout analysis utilities for local-memory op dispatches.

Provides functions for analyzing TileLayout thread/local partitions,
computing local region info, layout signature comparison, and thread
variable resolution. Used by cast.py, unary.py, and binary.py.
"""

import functools
import operator
from collections import defaultdict

from tvm.arith import Analyzer
from tvm.tirx.layout import TileLayout


def get_sublayout_from_region(layout, buffer_shape, region_st, region_extent):
    """Get sublayout by slicing the layout with the buffer region.

    Args:
        layout: The buffer's TileLayout.
        buffer_shape: The buffer's shape.
        region_st: Region start indices.
        region_extent: Region extents.

    Returns:
        Sublayout if slicing succeeds, otherwise the original layout.
    """
    if not layout:
        return layout
    region = [(region_st[i], region_st[i] + region_extent[i]) for i in range(len(region_st))]
    sliced = layout.slice(list(buffer_shape), region)
    return sliced if sliced is not None else layout


def get_layout_thread_local_partition(layout):
    """Extract thread and local dimension info from layout.

    Returns:
        tuple | None: On success, (thread_groups, local_dim_indices, local_extents).
            - thread_groups: dict {axis: (dim_indices, extents)} for each thread axis
            - local_dim_indices: list of dimension indices for local (memory) axes
            - local_extents: list of extents for local dimensions
            Returns None if layout is not supported.

    Validates:
        - No stride==0 on thread dims (broadcast/overlap = cross-thread semantics)
        - Local dims may have arbitrary strides (alignment uses actual layout strides)
        - No thread axes in replica

    Example:
        Layout (2, 8, 4, 2):(2@warpid, 4@laneid, 1@laneid, 1@m) returns:
        - thread_groups = {warpid: ([0], [2]), laneid: ([1, 2], [8, 4])}
        - local_dim_indices = [3], local_extents = [2]
    """
    if not isinstance(layout, TileLayout):
        return None

    shard = getattr(layout, "shard", None)
    if not shard:
        return None

    # Partition dimensions into thread and local (memory) axes
    thread_dim_indices = [i for i, it in enumerate(shard) if it.axis.is_thread()]
    local_dim_indices = [i for i, it in enumerate(shard) if not it.axis.is_thread()]

    if not thread_dim_indices or not local_dim_indices:
        return None

    analyzer = Analyzer()
    for idx in thread_dim_indices:
        if analyzer.can_prove_equal(shard[idx].stride, 0):
            return None

    # Replica must not contain thread axes
    replica = getattr(layout, "replica", None)
    if replica and any(it.axis.is_thread() for it in replica):
        return None

    # Group thread dimensions by axis
    thread_groups_dict = defaultdict(list)
    for idx in thread_dim_indices:
        thread_groups_dict[shard[idx].axis].append(idx)

    thread_groups = {}

    for axis, dim_indices in thread_groups_dict.items():
        dim_indices = sorted(dim_indices)
        extents = [shard[i].extent for i in dim_indices]
        thread_groups[axis] = (dim_indices, extents)

    local_extents = [shard[i].extent for i in local_dim_indices]
    return (thread_groups, local_dim_indices, local_extents)


def cast_layout_supported_for_local(layout) -> bool:
    """Check that layout is valid for local cast (warp/warpgroup/cta/cluster):
    filter out cross-thread semantics."""
    return get_layout_thread_local_partition(layout) is not None


def get_local_region(orig_layout: TileLayout, buffer_shape, region_st, region_extent):
    """Compute local storage shape, iteration starts, and extents with validation of region.

    Args:
        orig_layout: The original (unsliced) TileLayout.
        buffer_shape: The buffer shape.
        region_st: Region start in shape space.
        region_extent: Region extent in shape space.

    Returns:
        (local_shape, local_st, local_ext), or ([1], [0], [1]) if no local dims.
        Returns None if the region is invalid (non-contiguous slicing).
        - local_shape: full storage extents per local dim.
        - local_st: region start per local dim.
        - local_ext: region extent per local dim.

    Example:
        Layout (2, 8, 4, 2):(8@m, 2@laneid, 2@m, 1@m), Shape [16, 8], Region [8:16, :] returns:
        - local_shape = [2, 8], local_st = [1, 0], local_ext = [1, 8]
    """
    grouped, seps = orig_layout.group(list(buffer_shape))

    local_shape = []
    local_st = []
    local_ext = []
    analyzer = Analyzer()

    for d in range(len(buffer_shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        has_local = any(not grouped.shard[s].axis.is_thread() for s in shard_range)
        if not has_local:
            continue

        has_thread = any(grouped.shard[s].axis.is_thread() for s in shard_range)

        if not has_thread:
            # Pure local shape dim: use shape-level values directly.
            local_shape.append(buffer_shape[d])
            local_st.append(region_st[d])
            local_ext.append(region_extent[d])
        else:
            # Decompose start element
            remaining_st = region_st[d]
            st_coords = []
            for i, s_idx in enumerate(shard_range):
                sub_prod = 1
                for j in range(i + 1, len(shard_range)):
                    sub_prod = sub_prod * grouped.shard[shard_range[j]].extent
                st_coords.append(remaining_st // sub_prod)
                remaining_st = remaining_st % sub_prod

            # Decompose end element
            remaining_end = region_st[d] + region_extent[d] - 1
            end_coords = []
            for i, s_idx in enumerate(shard_range):
                sub_prod = 1
                for j in range(i + 1, len(shard_range)):
                    sub_prod = sub_prod * grouped.shard[shard_range[j]].extent
                end_coords.append(remaining_end // sub_prod)
                remaining_end = remaining_end % sub_prod

            # check the rectangularity and contiguity of the sliced region
            cur_local_shape, cur_local_st, cur_local_end = 1, 0, 0
            for k in reversed(range(len(st_coords))):
                if grouped.shard[seps[d] + k].axis.is_thread():
                    # for thread dims, region must be contiguous and span full extent
                    if not (
                        analyzer.can_prove_equal(st_coords[k], 0)
                        and analyzer.can_prove_equal(
                            end_coords[k], grouped.shard[seps[d] + k].extent - 1
                        )
                    ):
                        return None
                else:
                    if not analyzer.can_prove_equal(end_coords[k] - st_coords[k], 1) and not (
                        analyzer.can_prove_equal(st_coords[k], 0)
                        and analyzer.can_prove_equal(
                            end_coords[k], grouped.shard[seps[d] + k].extent - 1
                        )
                    ):
                        # to ensure contiguity, if the region spans multiple values
                        # in this dim, it must span the full extent
                        return None
                    cur_local_shape *= grouped.shard[seps[d] + k].extent
                    cur_local_st = cur_local_st * grouped.shard[seps[d] + k].extent + st_coords[k]
                    cur_local_end = (
                        cur_local_end * grouped.shard[seps[d] + k].extent + end_coords[k]
                    )

            # double check the validity of the sliced region
            assert region_extent[d] == functools.reduce(
                operator.mul, [end - st + 1 for st, end in zip(st_coords, end_coords)], 1
            )

            # append the local info without thread dims
            local_shape.append(cur_local_shape)
            local_st.append(cur_local_st)
            local_ext.append(cur_local_end - cur_local_st + 1)

    if not local_shape:
        return [1], [0], [1]  # treat no local dim case as 1D local shape with 1 element
    return local_shape, local_st, local_ext


def compute_linear_offset(region_st, local_dims, layout):
    """Compute linear offset using layout's actual strides.

    Physical offset = sum(region_st[dim] * layout.shard[dim].stride) for all local dims.
    """
    offset = 0
    for dim_idx in local_dims:
        offset = offset + region_st[dim_idx] * layout.shard[dim_idx].stride
    return offset


def _axis_key(axis):
    if hasattr(axis, "name") and axis.name:
        return str(axis.name)
    return str(axis)


def layout_signature(layout):
    """Return semantic signature from canonicalized TileLayout.

    Returns (thread_sig, local_sig, replica_sig).
    Each sig is a list of (axis_key, extent, stride) in shard/replica order.
    """
    if not isinstance(layout, TileLayout):
        return None
    shard = getattr(layout, "shard", None)
    if not shard:
        return None

    thread_sig = []
    local_sig = []
    for it in shard:
        item = (_axis_key(it.axis), it.extent, it.stride)
        if it.axis.is_thread():
            thread_sig.append(item)
        else:
            local_sig.append(item)

    replica_sig = []
    replica = getattr(layout, "replica", None) or []
    for it in replica:
        replica_sig.append((_axis_key(it.axis), it.extent, it.stride))
    return (thread_sig, local_sig, replica_sig)


def sig_equal(analyzer: Analyzer, src_sig, dst_sig) -> bool:
    """Compare two layout signatures with semantic equality (Analyzer).

    Signatures come from layout_signature(layout) and are:
      (thread_sig, local_sig, replica_sig)
    Each sig element is (axis_key, extent, stride).
    """
    if src_sig is None or dst_sig is None:
        return False

    src_thread_sig, src_local_sig, src_replica_sig = src_sig
    dst_thread_sig, dst_local_sig, dst_replica_sig = dst_sig

    if len(src_thread_sig) != len(dst_thread_sig):
        return False
    if len(src_local_sig) != len(dst_local_sig):
        return False
    if len(src_replica_sig) != len(dst_replica_sig):
        return False

    def _list_equal(a_list, b_list) -> bool:
        for (a_key, a_ext, a_str), (b_key, b_ext, b_str) in zip(a_list, b_list):
            if a_key != b_key:
                return False
            if not analyzer.can_prove_equal(a_ext, b_ext):
                return False
            if not analyzer.can_prove_equal(a_str, b_str):
                return False
        return True

    return (
        _list_equal(src_thread_sig, dst_thread_sig)
        and _list_equal(src_local_sig, dst_local_sig)
        and _list_equal(src_replica_sig, dst_replica_sig)
    )


def resolve_thread_var(axis, sctx):
    """Map the axis to the corresponding thread variable."""
    axis_name = getattr(axis, "name", None)
    if not axis_name:
        try:
            axis_name = str(axis)
        except Exception:
            axis_name = ""

    for key, itervar in sctx.launch_params.items():
        if getattr(itervar.var, "name", "") == axis_name:
            return itervar.var

    if axis_name:
        axis_name_lower = axis_name.lower()
        for key in sctx.launch_params:
            if axis_name_lower in key.lower() or (axis_name == "tx" and "threadIdx.x" in key):
                return sctx.launch_params[key].var

    if "threadIdx.x" in sctx.launch_params:
        return sctx.launch_params["threadIdx.x"].var

    return None
