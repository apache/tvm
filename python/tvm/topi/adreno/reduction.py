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
# pylint: disable=invalid-name,unused-variable,too-many-locals,len-as-condition
"""Schedule for reduce operators"""
import numpy
from tvm import te
from ..utils import get_const_tuple
from .injective import schedule_injective_from_existing
from .utils import get_div
from ..cuda.reduction import schedule_reduce_impl


def _schedule_reduce_adreno(op, sch, is_idx_reduce=False):
    sch_output = sch.outputs[0].output(0)
    use_rfactor = False
    if not is_idx_reduce:
        rdomain = 1
        whole_rop_output = op.output(0)
        for axis in sch[whole_rop_output].op.reduce_axis:
            rdomain = rdomain * axis.dom.extent
        if rdomain > 50:
            use_rfactor = True
            # shared goves better perf, but works only for rfactor flow
            scope = "shared"
        else:
            # in case of direct scheduling, shared is failed to be compiled
            scope = "local"
        if op in sch.outputs:
            whole_rop_output = sch.cache_write(sch_output, scope)
        else:
            # no change for whole_rop_output def, but need to set proper scope
            sch[whole_rop_output].set_scope(scope)
    else:
        temp_idx_input = op.input_tensors[0].op.output(0)
        temp_val_input = op.input_tensors[0].op.output(1)
        sch[temp_idx_input].set_scope("local")
        sch[temp_val_input].set_scope("local")

    shape = get_const_tuple(sch_output.shape)
    latest4 = len(shape) > 0 and shape[-1] == 4
    div4 = numpy.prod(shape) % 4 == 0

    # Fuse and split the axis
    if latest4:
        fused_outer = sch[sch_output].fuse(
            *[sch[sch_output].op.axis[i] for i in range(len(sch[sch_output].op.axis) - 1)]
        )
    else:
        fused_outer = sch[sch_output].fuse(
            *[sch[sch_output].op.axis[i] for i in range(len(sch[sch_output].op.axis))]
        )

    ftc = numpy.prod(shape)
    a = fused_outer

    if not is_idx_reduce:
        if use_rfactor:
            # below values were selected empirically assuming that we should have some work in each
            # thread (currently from 25-49) and number of threads not exceeding some threshold that
            # was selected as 256 from performance point of view after experiments on Adreno 660
            max_threads = rdomain.value // 25 if rdomain > 25 else 1
            max_threads = 256 if max_threads > 256 else max_threads
            num_thread = get_div(rdomain, max_threads)

            fused_reduce = sch[whole_rop_output].fuse(*sch[whole_rop_output].op.reduce_axis)
            thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
            _, ki = sch[whole_rop_output].split(fused_reduce, factor=num_thread)
            data_out_rf = sch.rfactor(whole_rop_output, ki)
            sch[data_out_rf].compute_at(
                sch[whole_rop_output], sch[whole_rop_output].op.reduce_axis[0]
            )
            sch[whole_rop_output].bind(sch[whole_rop_output].op.reduce_axis[0], thread_y)

    if div4:
        if latest4:
            b = sch[sch_output].op.axis[-1]
        else:
            a, b = sch[sch_output].split(fused_outer, factor=4)
        sch[sch_output].vectorize(b)
        if not use_rfactor:
            if is_idx_reduce:
                sch[temp_idx_input].compute_at(sch[sch_output], b)
                sch[temp_val_input].compute_at(sch[sch_output], b)
            else:
                sch[whole_rop_output].compute_at(sch[sch_output], b)

    if not use_rfactor:
        num_thread = get_div(ftc, 128)
        bx, outer_in = sch[sch_output].split(a, factor=num_thread)
        sch[sch_output].bind(bx, te.thread_axis("blockIdx.x"))
        sch[sch_output].bind(outer_in, te.thread_axis("threadIdx.x"))

        if not div4:
            if is_idx_reduce:
                sch[temp_idx_input].compute_at(sch[sch_output], outer_in)
                sch[temp_val_input].compute_at(sch[sch_output], outer_in)
            else:
                sch[whole_rop_output].compute_at(sch[sch_output], outer_in)
    else:
        sch[sch_output].bind(a, te.thread_axis("blockIdx.x"))
        if not div4 or use_rfactor:
            if is_idx_reduce:
                sch[temp_idx_input].compute_at(sch[sch_output], a)
                sch[temp_val_input].compute_at(sch[sch_output], a)
            else:
                sch[whole_rop_output].compute_at(sch[sch_output], a)


def schedule_reduce(outs):
    return schedule_reduce_impl(
        outs, _schedule_reduce_adreno, schedule_injective_from_existing, True
    )
