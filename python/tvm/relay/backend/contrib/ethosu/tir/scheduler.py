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
# pylint: disable=invalid-name, unused-argument
"""Scheduling for Arm(R) Ethos(TM)-U NPU."""
import tvm


def schedule(cached_func, const_dict, cascader=None):
    """Schedule a CachedFunc for NPU compilation.

    Parameters
    ----------
    cached_func : CachedFunc
        The CachedFunc to schedule.
    const_dict : dict of int to numpy.ndarray
        The constant dictionary.
    cascader : callable, optional
        A cascading function to apply optimizing scheduling
        to the graph.

    Returns
    -------
    s : tvm.te.Schedule
        The completed schedule for the graph.

    """
    s = tvm.te.create_schedule([t.op for t in cached_func.outputs])
    if cascader:
        cascader(cached_func, const_dict, s)
    inline_no_ops(cached_func, s)
    schedule_pragmas(s)
    schedule_cache_reads(s)
    return s


def tile_nd(s, tensor, tile):
    """Scheduling utility to perform N-dimensional tiling.

    Parameters
    ----------
    s : tvm.te.Schedule
        The schedule to apply the tiling to.
    tensor : tvm.te.Tensor
        The tensor to apply the tiling to.
    tile : tuple
        The N-dimensional tile size.

    Returns
    -------
    outer_indices : list of tvm.tir.IterVar
        The outer iteration variables.
    inner_indices : list of tvm.tir.IterVar
        The inner iteration variables.

    """
    outer_indices = []
    inner_indices = []
    for i, size in enumerate(tile):
        outer, inner = s[tensor].split(tensor.op.axis[i], size)
        outer_indices.append(outer)
        inner_indices.append(inner)

    s[tensor].reorder(*outer_indices, *inner_indices)
    return outer_indices, inner_indices


def total_cascader(stripe_size):
    """A demo/test cascader which tries to cascade every op in the graph together.

    The desired output stride size should be specified. Note this only works
    for single output graphs.

    Parameters
    ----------
    stripe_size : tuple
        The output stripe size.

    Returns
    -------
    func : callable
        The cascading function.

    """

    def _cascader(cached_func, const_dict, sch):
        scheduled = set()

        def _visit(tensor, stage, ax):
            if tensor not in scheduled and isinstance(tensor.op, tvm.te.ComputeOp):
                sch[tensor].compute_at(stage, ax)
                scheduled.add(tensor)
                for input_tensor in tensor.op.input_tensors:
                    _visit(input_tensor, stage, ax)

        assert len(cached_func.outputs) == 1
        out = cached_func.outputs[0]
        oi, _ = tile_nd(sch, out, stripe_size)
        for ax in oi:
            sch[out].unroll(ax)
        for input_tensor in out.op.input_tensors:
            _visit(input_tensor, sch[out], oi[-1])

    return _cascader


def copy_constants():
    """A simple planner which copies all constant data from FLASH -> SRAM.

    Returns
    -------
    planner : callable
        The planning function.
    """

    def _planner(cached_func, const_dict, sch):
        planned = set()  # type: ignore

        def _visit(tensor, reader):
            if tensor is not planned:
                planned.add(tensor)
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    index = list(cached_func.inputs).index(tensor)
                    if index in const_dict:
                        sch.cache_read(tensor, "global", [reader])

                elif isinstance(tensor.op, tvm.te.ComputeOp):
                    for input_tensor in tensor.op.input_tensors:
                        _visit(input_tensor, tensor)

        for output_tensor in cached_func.outputs:
            _visit(output_tensor, None)

    return _planner


def schedule_pragmas(sch):
    """Add pragmas to the operators that require them.

    This adds the pragmas used for codegen to the NPU ops.
    They are taken directly from the TE compute op's attributes.
    Modifies the schedule in-place.

    Parameters
    ----------
    sch : tvm.te.Schedule
        The schedule.

    """

    def _add_pragmas(stage, ax):
        if "op" in [attr for attr, val in stage.op.attrs.items()]:
            stage.pragma(ax, "op", stage.op.attrs["op"])
            for attr, val in stage.op.attrs.items():
                if attr != "op":
                    stage.pragma(ax, str(attr), val)

    for stage in sch.stages:
        if (
            isinstance(stage.op, tvm.te.ComputeOp)
            and len(stage.op.axis) + len(stage.op.reduce_axis) > 0
        ):
            # The logic ensures the pragmas are assigned to the inner tiling loops
            # rather than the outer ones (which end up getting unrolled).
            num_inner_loops = len(stage.op.axis) + len(stage.op.reduce_axis)
            ax = stage.leaf_iter_vars[-num_inner_loops]
            _add_pragmas(stage, ax)


def schedule_cache_reads(sch):
    """Schedule cache reads that have been introduced.

    There are two things we need to happen to cache_read stages. They should be tagged
    with the 'ethosu_copy' pragma and have all their axes fused to make them 1D.

    Parameters
    ----------
    sch : tvm.te.Schedule
        The schedule.

    """

    def _detect_cache_read(stage):
        # Try and detect cache_reads by checking if the compute op is identity
        if isinstance(stage.op, tvm.te.ComputeOp):
            op = stage.op
            if "ethosu" in op.name:
                return False
            axes = op.axis
            if len(op.input_tensors) == 1:
                tensor = op.input_tensors[0]
                try:
                    identity_op = tensor(*axes)
                except ValueError:
                    return False
                if tvm.tir.analysis.expr_deep_equal(identity_op, op.body[0]):
                    return True
        return False

    for stage in sch.stages:
        if _detect_cache_read(stage):
            fax = stage.fuse(*stage.op.axis)
            stage.pragma(fax, "op", "ethosu_copy")


def inline_no_ops(cached_func, sch):
    """Inline 'no-ops' - operations that in principle do nothing.

    Modifies the schedule in-place. For now we inline reshape and
    strided slice - more could be added.

    Parameters
    ----------
    cached_func : CachedFunc
        The cached func.
    sch : tvm.te.Schedule
        The schedule.

    """
    no_ops = {"T_reshape", "T_strided_slice"}
    scheduled = set()

    def _visit(tensor):
        if tensor not in scheduled and isinstance(tensor.op, tvm.te.ComputeOp):
            if tensor.op.name in no_ops:
                sch[tensor].compute_inline()
            scheduled.add(tensor)
            for input_tensor in tensor.op.input_tensors:
                _visit(input_tensor)

    for out in cached_func.outputs:
        _visit(out)


class Convolution2DCompute:
    """A helper class to manipulate the series of compute ops that make up a 2D convolution."""

    def __init__(self, read, convert_to_nhwc, pad, conv2d, convert_to_nhcwb16, write):
        self.read = read
        self.convert_to_nhwc = convert_to_nhwc
        self.pad = pad
        self.conv2d = conv2d
        self.convert_to_nhcwb16 = convert_to_nhcwb16
        self.write = write

    @classmethod
    def from_output(cls, out):
        write = out
        convert_to_nhcwb16 = write.op.input_tensors[0]
        conv2d = convert_to_nhcwb16.op.input_tensors[0]
        pad = conv2d.op.input_tensors[0]
        convert_to_nhwc = pad.op.input_tensors[0]
        read = convert_to_nhwc.op.input_tensors[0]
        return cls(read, convert_to_nhwc, pad, conv2d, convert_to_nhcwb16, write)

    def split(self, sch, axis, val):
        outer, inner = sch[self.write].split(self.write.op.axis[axis], val)
        sch[self.write].reorder(
            outer, *[ax for ax in self.write.op.axis if ax != self.write.op.axis[axis]], inner
        )
        sch[self.write].unroll(outer)
        g = sch.create_group(outputs=self.convert_to_nhcwb16, inputs=self.read, include_inputs=True)
        g.compute_at(sch[self.write], outer)
        return outer
