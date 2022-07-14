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
import io

import tvm
from tvm import topi, IRModule
import numpy as np
from tvm.contrib import utils, clang
import tvm.testing
from tvm import te
from typing import Union


def _create_schedule(
    placeholder: list,
    c_code: Union[str, io.TextIOWrapper] = "",
    use_external_conv2d_impl: bool = True,
):
    # How to do the same with TE
    # Add pragma TE
    # s = te.create_schedule(result.op)
    # axis = result.op.axis
    # s[result].pragma(axis[0], "import_llvm", c_to_llvm())
    # with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, my_ai_hw_conv2d_pass)]}):
    #     mod = tvm.lower(s, [ifmap, weights, result], simple_mode=True)
    #
    # llvm_mod = tvm.build(mod, [ifmap, weights, result], target=target, name="test_external_conv2d")
    # llvm_mod(ifmap_data, weight_data, result_data)
    if isinstance(c_code, io.TextIOWrapper):
        c_code_str = c_code.read()
    elif isinstance(c_code, str):
        c_code_str = c_code
    else:
        raise TypeError()

    assert (
        use_external_conv2d_impl
        and c_code_str != ""
        or not use_external_conv2d_impl
        and c_code_str == ""
    )

    def _c_to_llvm(c_code: str) -> str:
        temp = utils.tempdir()
        ll_path = temp.relpath("conv2d.ll")
        ll_code = clang.create_llvm([c_code], output=ll_path)
        return ll_code

    func_tir = te.create_prim_func(placeholder)
    ir_module_from_te = IRModule({"main": func_tir})
    sch_tir = tvm.tir.Schedule(ir_module_from_te)
    if use_external_conv2d_impl:
        conv2d_b = sch_tir.get_block("conv2d_nchw")
        conv2d_l = sch_tir.get_loops(conv2d_b)
        sch_tir.annotate(conv2d_l[0], "pragma_import_llvm", _c_to_llvm(c_code_str))
    return sch_tir


def _generate_io_arrays(shapes: dict, dev):
    n, w, h, ci, kw, kh, co = (
        shapes["n"],
        shapes["w"],
        shapes["h"],
        shapes["ci"],
        shapes["kw"],
        shapes["kh"],
        shapes["co"],
    )

    ifmap_data = tvm.nd.array(np.random.uniform(size=(n, ci, w, h)).astype("float32"), dev)
    weight_data = tvm.nd.array(np.random.uniform(size=(co, ci, kh, kw)).astype("float32"), dev)
    result_data = tvm.nd.array(np.zeros((n, co, w, h)).astype("float32"), dev)
    return ifmap_data, weight_data, result_data
