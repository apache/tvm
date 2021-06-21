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
"""Implement the official BYOC partitioning flow for Android NNAPI."""
import tvm
import tvm.relay
import tvm.relay.op.contrib.register
from . import _base

# avoid re-registering byoc annotation rules
_BYOC_ANNOTATION_RULES_REGISTERED = False


def _register_byoc_annotation_rules(external_compiler, android_nnapi_level):
    global _BYOC_ANNOTATION_RULES_REGISTERED
    # avoid re-registering byoc annotation rules
    if _BYOC_ANNOTATION_RULES_REGISTERED:
        return
    _BYOC_ANNOTATION_RULES_REGISTERED = True

    from tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter import (  # pylint: disable=import-outside-toplevel
        convert_relayir_to_nnapi,
    )
    from tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.error import (  # pylint: disable=line-too-long,import-outside-toplevel
        AndroidNNAPICompilerIncompatibleError,
    )
    import tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.operation_utils.relay_op as relay_op_handler_root  # pylint: disable=line-too-long,import-outside-toplevel

    def _isolate_op_call_node(call, compiler):
        func_params = []
        new_call_args = []
        for i, arg in enumerate(call.args):
            if isinstance(arg.checked_type, tvm.relay.TupleType):
                tuple_param_fields = [
                    tvm.relay.var(f"arg{ i }.{ j }", type_annotation=f)
                    for j, f in enumerate(arg.checked_type.fields)
                ]
                func_params += tuple_param_fields
                tuple_arg = tvm.relay.Tuple(
                    [tvm.relay.annotation.compiler_begin(f, compiler) for f in tuple_param_fields]
                )
                new_call_args.append(tuple_arg)
            elif isinstance(arg.checked_type, tvm.relay.TensorType):
                func_params.append(tvm.relay.var(f"arg{ i }", type_annotation=arg.checked_type))
                new_call_args.append(tvm.relay.annotation.compiler_begin(func_params[-1], compiler))
            else:
                raise NotImplementedError(arg.checked_type)
        new_call = tvm.relay.annotation.compiler_end(
            tvm.relay.Call(call.op, new_call_args, call.attrs, call.type_args), compiler
        )
        return tvm.relay.Function(func_params, new_call)

    def _check_call_support(call):
        assert isinstance(call, tvm.relay.Call)
        mod = tvm.IRModule.from_expr(_isolate_op_call_node(call, external_compiler))
        mod = tvm.relay.transform.PartitionGraph()(mod)
        mod, _ = _base.post_partition_transform(
            mod, {}, android_nnapi_level=android_nnapi_level, external_compiler=external_compiler
        )
        external_func = (lambda op: op if isinstance(op, tvm.relay.Function) else mod[op])(
            mod["main"].body.op
        )  # op may be a GlobalVar, hence the if
        assert isinstance(external_func, tvm.relay.Function)
        external_func = external_func.with_attr(
            "NnapiClassName", f"{ external_func.attrs.global_symbol }_0"
        )  # NnapiClassName is required for the converter
        try:
            convert_relayir_to_nnapi(external_func)
        except AndroidNNAPICompilerIncompatibleError:
            return False
        return True

    tvm.ir.register_op_attr(
        "annotation.compiler_begin", "target.android_nnapi", lambda expr: False
    )  # create "target.android_nnapi" in OpAttrMap

    def _recursive_register(cur_namespace, handle):
        if (
            hasattr(handle, "handler")
            and tvm.relay.op.get(cur_namespace).get_attr("target.android_nnapi") is None
        ):  # avoid overriding user-registered rules
            tvm.ir.register_op_attr(cur_namespace, "target.android_nnapi", _check_call_support)
        else:  # non-leaf
            for attr_name in dir(handle):
                if not attr_name.startswith("_"):
                    _recursive_register(
                        f"{ cur_namespace }.{ attr_name }" if cur_namespace != "" else attr_name,
                        getattr(handle, attr_name),
                    )

    _recursive_register("", relay_op_handler_root)


def _prune_android_nnapi_subgraphs(mod, external_compiler):
    """Prune a IRModule for subgraphs that are not suitable to be offloaded
    to Android NNAPI.

    Parameters
    ----------
    mod: tvm.IRModule
        The TVM Module to be pruned.

    external_compiler: str
        The name of the Android NNAPI external compiler.

    Returns
    -------
    mod: tvm.IRModule
        The pruned TVM Module.
    """

    def _func_should_be_pruned(func):
        if getattr(func.attrs, "Compiler", None) != external_compiler:
            return False

        def _scope():
            visitor = tvm.relay.ExprVisitor()
            visitor.visit(func)
            return {expr for expr in visitor.memo_map if isinstance(expr, tvm.ir.Op)}

        all_ops = _scope()
        if any([wanted_op in all_ops for wanted_op in [tvm.relay.op.get("nn.conv2d")]]):
            return False
        return True

    subgraphs_to_prune = {
        gv.name_hint for gv in mod.get_global_vars() if _func_should_be_pruned(mod[gv])
    }
    if len(subgraphs_to_prune) == 0:
        return mod

    def _remove_subgraphs(mod, subgraphs_to_prune):
        class InlineSubgraphs(tvm.relay.ExprMutator):
            """Inline subgraphs back to the invocation place."""

            def __init__(self, subgraphs_to_prune):
                super().__init__()
                self._subgraphs_to_prune = subgraphs_to_prune

            def __call__(self, mod):
                self._mod = mod
                new_mod = tvm.IRModule()
                gvs = mod.get_global_vars()
                for gvar in gvs:
                    new_mod[gvar] = self.visit(mod[gvar])
                return new_mod

            def visit_call(self, call):
                if (
                    isinstance(call.op, tvm.relay.GlobalVar)
                    and call.op.name_hint in self._subgraphs_to_prune
                ):
                    gfunc = self._mod[call.op]
                    bind_map = {}
                    assert len(gfunc.params) == len(call.args)
                    for i in range(len(call.args)):
                        bind_map[gfunc.params[i]] = self.visit(call.args[i])
                    return tvm.relay.bind(gfunc.body, bind_map)
                return super().visit_call(call)

        mod = InlineSubgraphs(subgraphs_to_prune)(mod)
        return tvm.IRModule(
            {gv: mod[gv] for gv in mod.get_global_vars() if gv.name_hint not in subgraphs_to_prune}
        )

    return _remove_subgraphs(mod, subgraphs_to_prune)


def byoc_partition(mod, params, android_nnapi_level):
    """Partition a IRModule using rules registered with TVM BYOC.

    Parameters
    ----------
    mod: tvm.IRModule
        The TVM Module to be partitioned.

    params: dict of str to tvm.runtime.NDArray
        The parameters to mod.

    android_nnapi_level: int
        The targeted Android API level.

    Returns
    -------
    mod: tvm.IRModule
        The partitioned module.

    params: dict of str to tvm.runtime.NDArray
        The transformed parameters to mod.
    """
    assert isinstance(mod, tvm.IRModule)

    external_compiler = "android_nnapi"
    _register_byoc_annotation_rules(external_compiler, android_nnapi_level)
    pattern_table = tvm.relay.op.contrib.register.get_pattern_table(external_compiler)
    if pattern_table is not None:
        mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget([external_compiler])(mod)
    mod = tvm.relay.transform.MergeCompilerRegions()(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = _prune_android_nnapi_subgraphs(mod, external_compiler)
    mod, params = _base.post_partition_transform(mod, params, android_nnapi_level)
    return mod, params
