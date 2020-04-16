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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
Relay TensorRT codegen.
"""
import tvm
from tvm import relay
from tvm.relay.expr import Call, Constant

from tvm.relay.transform import _ffi_api
from .expr_functor import ExprMutator

def _bind_params(func, params):
    """
    Bind the params to the expression as constants.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)

class LegalizeLayoutTranform(ExprMutator):
    """
    Legalize Relay layout transforms to transpose ops to simplify TensorRT conversion.
    """
    def visit_call(self, expr):
        visit = super().visit_call(expr)
        if expr.op == tvm.relay.op.get("layout_transform"):
            src_layout = expr.attrs['src_layout']
            dst_layout = expr.attrs['dst_layout']
            if src_layout == "NCHW" and dst_layout == "NHWC":
                return relay.transpose(visit.args[0], axes=[0, 2, 3, 1])
            elif src_layout == "NHWC" and dst_layout == "NCHW":
                return relay.transpose(visit.args[0], axes=[0, 3, 1, 2])
            elif src_layout == "HWIO" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[3, 2, 0, 1])
            elif src_layout == "HWOI" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
            elif src_layout == "HWIO" and dst_layout == "IOHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
        return visit

class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """
    def visit_tuple_getitem(self, expr):
        visit = super().visit_tuple_getitem(expr)
        if visit.index != 0:
            return visit
        elif isinstance(visit.tuple_value, Call) and visit.tuple_value.op.name == "nn.dropout":
            return visit.tuple_value.args[0]
        return visit

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, expr):
        if expr.op.name == "multiply":
            if isinstance(expr.args[1], Constant):
                data = expr.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return expr.args[0]
        return super().visit_call(expr)

class RemoveRedundantTranspose(ExprMutator):
    """
    Removes Transpose([1, 0]) followed by Transpose([1, 0]). This pass, when
    preceded by with RemoveMultiplyByOne is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def check_axes(self, axes):
        return len(axes) == 2 and int(axes[0].value) == 1 and int(axes[1].value) == 0

    def visit_call(self, expr):
        if expr.op.name == "transpose":
            if self.check_axes(expr.attrs['axes']):
                if isinstance(expr.args[0], Call) and expr.args[0].op.name == "transpose":
                    if self.check_axes(expr.args[0].attrs['axes']):
                        return expr.args[0].args[0]
        return super().visit_call(expr)

def PreprocessForTrt(mod):
    """Applies passes to prepare main function for TensorRT conversion.

    Parameters
    ----------
    mod: Module
        The original module.

    Returns
    -------
    mod: Module
        The module modified for TensorRT.
    """
    mod['main'] = LegalizeLayoutTranform().visit(mod['main'])
    mod['main'] = RemoveDropout().visit(mod['main'])
    mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
    mod['main'] = RemoveRedundantTranspose().visit(mod['main'])
    return mod

def GetTrtVersion():
    """Gets the version of TensorRT that TVM is built against.

    Returns
    -------
    ret: Tuple[int]
        TensorRT version as a tuple of major, minor, and patch number. If TVM
        is not built with TensorRT, an empty tuple is returned instead.
    """
    return tuple(map(int, _ffi_api.GetTrtVersion()))

def IsTrtRuntimeAvailable():
    if not tvm.get_global_func("relay._transform.GetTrtVersion", True):
        return False
    return GetTrtVersion() != ()

def EnableTrt(mod, params=None, trt_version=None):
    """Converts the "main" function in the module into one that can be executed using
    TensorRT. If any of the operators are not supported by the TensorRT
    conversion, the unmodified program will be returned instead.

    Parameters
    ----------
    mod: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    trt_version : Optional[Tuple[int]]
        Which version of TensorRT to target for partitioning as a tuple of
        (major, minor, patch). If not specified, will attempt to get using
        GetTrtVersion.

    Returns
    -------
    mod: Module
        The modified module which will use the TensorRT runtime if compatible.
    """
    if not trt_version:
        trt_version = GetTrtVersion()
        # If TVM wasn't built against TRT, default to target TRT 6. Since the
        # actual conversion to TRT is done at runtime, building against TRT is
        # not required for compilation.
        if not trt_version:
            trt_version = (6, 0, 1)
    assert isinstance(trt_version, (list, tuple))
    assert len(trt_version) == 3

    # Apply passes required for TRT
    seq = relay.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                      relay.transform.ConvertLayout('NCHW')])
    with relay.transform.PassContext(opt_level=3):
        mod = seq(mod)
    mod = PreprocessForTrt(mod)
    if params:
        # Bind params so that we can use FoldConstant.
        mod['main'] = _bind_params(mod['main'], params)
    mod = relay.transform.FoldConstant()(mod)
    return _ffi_api.EnableTrt(*trt_version)(mod)
