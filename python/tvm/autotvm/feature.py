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
# pylint: disable=invalid-name,
"""Extract feature of iter vars

There are two types of feature
1) Itervar feature
   This feature is extracted based on loop variables.
   Different loop structures will result in different shapes of feature
2) Curve sample feature (relation feature)
   This feature is extracted by sampling relation curve.
   This feature is invariant of loop structure.
"""

import struct
import numpy as np
import tvm._ffi

from tvm.target import Target
from tvm.driver import build_module


def ana_lower(sch, args, binds=None, simple_mode=True):
    """Do lower while keeping all axes in IR
    i.e. Do not eliminate loop with extent of 1, do not vectorize, unroll or inject virtual threads
    """
    sch = sch.normalize()
    # Phase 0
    context = tvm.transform.PassContext(config={"tir.debug_keep_trivial_loop": True})
    with context:
        mod = build_module.schedule_to_module(sch, args, binds=binds)

    mod = tvm.tir.transform.StorageFlatten(64)(mod._move())
    mod = tvm.tir.transform.Simplify()(mod._move())
    assert simple_mode
    return mod["main"].body


try:
    _get_buffer_curve_sample_flatten = tvm._ffi.get_global_func(
        "autotvm.feature.GetCurveSampleFeatureFlatten"
    )
    _get_itervar_feature = tvm._ffi.get_global_func("autotvm.feature.GetItervarFeature")
    _get_itervar_feature_flatten = tvm._ffi.get_global_func(
        "autotvm.feature.GetItervarFeatureFlatten"
    )
except ValueError as e:

    def raise_error(*args, **kwargs):  # pylint: disable=unused-argument
        raise RuntimeError("Cannot load autotvm c++ API")

    _get_buffer_curve_sample_flatten = (
        _get_itervar_feature
    ) = _get_itervar_feature_flatten = raise_error


def get_itervar_feature(sch, args, take_log=False):
    """get features of iter vars

    Parameters
    ----------
    sch: tvm.te.schedule.Schedule
    args: Array of te.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    features of every axis in the IR, see doc/features.md for detail
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature(stmt, take_log)

    # convert tvm node to python type
    ret = []
    for row in feas:
        tmp = []
        tmp.append([row[0][0].value, row[0][1]])
        for item in row[1:]:
            tmp.append([item[0].value] + [x.value for x in item[1:]])
        ret.append(tmp)
    return ret


def flatten_itervar_feature(fea):
    """flatten features into one-dimensional feature vectors

    Parameters
    ----------
    fea: list
        return value of get_itervar_feature

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    flatten = []
    for axis in fea:
        for pair in axis[1:]:
            flatten.append(pair[1:])
    return np.concatenate(flatten)


def get_itervar_feature_flatten(sch, args, take_log=True):
    """get flatten features of iter vars
    this is equivalent to get_itervar_feature + flatten_itervar_feature, but much faster.

    Parameters
    ----------
    sch: tvm.te.schedule.Schedule
    args: Array of te.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature_flatten(stmt, take_log)
    feas = struct.unpack(f"{len(feas) // 4}f", feas)
    return feas


def get_flatten_name(fea):
    """Get names of feature after flatten.

    Parameters
    ----------
    fea: list or str
        return value of get_itervar_feature or a line of logfile

    Returns
    -------
    feature_names: Array of str
    """

    feature_name = {
        "_attr_": ["length", "nest_level", "topdown", "bottomup"] + [f"ann_{i}" for i in range(20)],
        "_arith_": ["add", "mul", "div"],
        "buf_touch": ["stride", "mod", "count", "reuse", "T_count", "T_reuse"],
    }

    if isinstance(fea, str):
        # pylint: disable=import-outside-toplevel
        from .record import decode

        # flatten line to feature
        line = fea
        ret = decode(line)
        if ret is None:
            raise ValueError("Unsupported AutoTVM log format")
        inp, _ = ret
        target = Target(inp.target)
        with target:
            s, args = inp.template.instantiate(inp.config)
        fea = get_itervar_feature(s, args)

    names = []
    ct = 0
    for row in fea:
        var_name = str(row[0][1])
        for pair in row[1:]:
            key = pair[0]
            if key in feature_name:
                name_list = feature_name[key]
            else:
                name_list = feature_name["buf_touch"]

            for i in range(len((pair[1:]))):
                names.append(".".join([f"f{ct}", var_name, key, name_list[i]]))
                ct += 1
    return names


def get_buffer_curve_sample_flatten(sch, args, sample_n=30):
    """
    Get flatten curve sample feature (relation feature)

    Parameters
    ----------
    sch: tvm.te.schedule.Schedule
    args: Array of te.tensor.Tensor
        the buffer args for lower
    sample_n: int
        number of sample points along one dimension

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_buffer_curve_sample_flatten(stmt, sample_n, False)
    feas = struct.unpack(f"{len(feas) // 4}f", feas)
    return feas
