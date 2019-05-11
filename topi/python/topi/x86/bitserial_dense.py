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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from topi.util import get_const_int
from .. import tag
from .. import generic

@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_dense, ['cpu'], 'direct')
def schedule_bitserial_dense(cfg, outs):
    """Schedule for bitserial_dense.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of bitserial dense operator.
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for bitserial_dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, data_vec, weight_vec, output):
        s[data_vec].parallel(s[data_vec].op.axis[0])
        s[weight_vec].parallel(s[weight_vec].op.axis[0])

        y, x = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)


        cfg["reorder_0"].apply(s, output, [yo, xo, ko, yi, wb, db, ki, xi])
        cfg["ann_reduce"].apply(s, output, [db, wb],
                                axis_lens=[get_const_int(db.dom.extent),
                                           get_const_int(wb.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg["ann_spatial"].apply(s, output, [yi, xi],
                                 axis_lens=[cfg['tile_y'].size[-1],
                                            cfg['tile_x'].size[-1]],
                                 max_unroll=8,
                                 cfg=cfg)
        s[output].vectorize(xi)
        s[output].parallel(yo)
        return s

    def traverse(op):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or 'elemwise' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        elif op.tag == 'bitserial_dense' or 'bitserial_dense_unipolar':
            output = op.output(0)
            weight_vec = op.input_tensors[0]

            data_vec = op.input_tensors[1]
            data = data_vec.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                data = data.op.input_tensors[0]
            _schedule(cfg, s, data_vec, weight_vec, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

    traverse(outs[0].op)
    return s
