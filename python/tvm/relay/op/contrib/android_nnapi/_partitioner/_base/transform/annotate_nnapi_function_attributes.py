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
"""Annotate Android NNAPI functions (in Relay IR) for additional
attributes required for lowering."""
import tvm
import tvm.relay


class AnnotateNnapiFunctionAttributes:
    """Tag Android NNAPI compiler-specific attributes to exported Relay IR Functions.

    Parameters
    ----------
    external_compiler: str
        The name of the BYOC external compiler.

    android_nnapi_level: int
        The targeted Android API level.
    """

    def __init__(self, external_compiler, android_nnapi_level):
        super().__init__()
        self._external_compiler = external_compiler
        self._android_nnapi_level = android_nnapi_level

    def __call__(self, mod):
        """Tag Android NNAPI compiler-specific attributes to exported Relay IR Functions.

        Parameters
        ----------
        mod: tvm.IRModule
            The module containing exported functions to be tagged.

        Returns
        -------
        mod: tvm.IRModule
            The tagged module.
        """
        assert isinstance(mod, tvm.IRModule)
        ret = tvm.IRModule()
        gvs = mod.get_global_vars()
        for gvar in gvs:
            func = mod[gvar]
            func = self._Annotator(self._external_compiler, self._android_nnapi_level).annotate(
                func
            )
            ret[gvar] = func
        return ret

    class _Annotator(tvm.relay.ExprMutator):
        def __init__(self, external_compiler, android_nnapi_level):
            super().__init__()
            self._external_compiler = external_compiler
            self._android_nnapi_level = android_nnapi_level

        def annotate(self, func):
            assert isinstance(func, tvm.relay.Function)
            return self.visit(func)

        def visit_function(self, fn):
            new_func = super().visit_function(fn)
            if getattr(new_func.attrs, "Compiler", None) == self._external_compiler:
                new_func = new_func.with_attr(
                    "NnapiTargetVersion", tvm.tir.IntImm("int32", self._android_nnapi_level)
                )
            return new_func
