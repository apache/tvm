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

import json

import pytest

import tvm.testing
from tvm.script import ir as I, tir as T


class Base:
    def test_metadata(self):
        """Validate the generated metadata string

        The metadata string should be valid JSON, which is parsed to
        the same structure as defined by a test cases `expected`.
        Comparison is done after parsing, so that the test case is
        agnostic to any pretty-printing done for the JSON string.
        """

        mod = self.mod
        mod = tvm.tir.transform.GenerateFunctionSignatureMetadata()(mod)

        func = mod[tvm.runtime.Module._GET_TIR_FUNCTION_METADATA]
        metadata_str = func.body.value.args[0].value

        metadata = json.loads(metadata_str)
        assert metadata == self.expected

    def test_metadata_function(self):
        """Validate the PrimFunc containing the metadata string"""

        mod = self.mod
        mod = tvm.tir.transform.GenerateFunctionSignatureMetadata()(mod)
        func = mod[tvm.runtime.Module._GET_TIR_FUNCTION_METADATA]
        metadata_str = func.body.value.args[0].value

        @T.prim_func
        def expected() -> T.handle:
            T.func_attr(
                {
                    "global_symbol": tvm.runtime.Module._GET_TIR_FUNCTION_METADATA,
                    "tir.is_host_func": True,
                }
            )
            return metadata_str

        tvm.ir.assert_structural_equal(func, expected)

    def test_no_other_changes_to_module(self):
        """Only change to IRModule is the new function

        All other functions are pass through unmodified.
        """
        mod = self.mod
        mod = tvm.tir.transform.GenerateFunctionSignatureMetadata()(mod)
        del mod[tvm.runtime.Module._GET_TIR_FUNCTION_METADATA]

        tvm.ir.assert_structural_equal(self.mod, mod)


class TestEmptyModule(Base):
    @property
    def mod(self):
        @I.ir_module
        class Module:
            pass

        return Module

    expected = {
        "functions": {},
    }


class TestSingleFunction(Base):
    @property
    def mod(self):
        @I.ir_module
        class Module:
            @T.prim_func
            def main():
                pass

        return Module

    expected = {
        "functions": {
            "main": {
                "params": [],
            },
        },
    }


class TestMultipleFunctions(Base):
    @property
    def mod(self):
        @I.ir_module
        class Module:
            @T.prim_func
            def func_a():
                pass

            @T.prim_func
            def func_b():
                pass

        return Module

    expected = {
        "functions": {
            "func_a": {
                "params": [],
            },
            "func_b": {
                "params": [],
            },
        },
    }


class TestPrivateFunction(Base):
    """Private functions should not be exposed externally"""

    @property
    def mod(self):
        @I.ir_module
        class Module:
            @T.prim_func
            def func_a():
                pass

            @T.prim_func(private=True)
            def func_b():
                pass

        return Module

    expected = {
        "functions": {
            "func_a": {
                "params": [],
            },
        },
    }


@pytest.mark.xfail(reason="Not yet implemented")
class TestPrimitiveArguments(Base):
    """Annotation of primitive arguments"""

    @property
    def mod(self):
        @I.ir_module
        class Module:
            @T.prim_func
            def func_a(A: T.int32, B: T.float16):
                pass

        return Module

    expected = {
        "functions": {
            "func_a": {
                "params": [
                    {
                        "name": "A",
                        "type": "PrimType",
                        "dtype": "int32",
                    },
                    {
                        "name": "A",
                        "type": "PrimType",
                        "dtype": "float16",
                    },
                ],
            },
        },
    }


if __name__ == "__main__":
    tvm.testing.main()
