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
# pylint: disable=missing-docstring
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.relax.transform import AttachExternModules


def _infer_scalar_add(x, y):  # pylint: disable=invalid-name
    assert isinstance(x, nn.Tensor)
    assert isinstance(y, nn.Tensor)
    assert x.ndim == 0 and x.dtype == "float32"
    assert y.ndim == 0 and y.dtype == "float32"
    return nn.Tensor.placeholder(shape=(), dtype="float32")


def _infer_test_sym(a, b):  # pylint: disable=invalid-name
    def _var_equal(a, b):  # pylint: disable=invalid-name
        return tvm.ir.structural_equal(a, b, map_free_vars=True)

    assert isinstance(a, nn.Tensor)
    assert isinstance(b, nn.Tensor)
    assert a.ndim == 3 and a.dtype == "float32"  # [x, y, 1]
    assert b.ndim == 3 and b.dtype == "float32"  # [y, z, 5]
    x, y, z = a.shape[0], b.shape[0], b.shape[1]  # pylint: disable=invalid-name
    assert _var_equal(a.shape[0], x)
    assert _var_equal(a.shape[1], y)
    assert a.shape[2] == 1
    assert _var_equal(b.shape[0], y)
    assert _var_equal(b.shape[1], z)
    assert b.shape[2] == 5
    return nn.Tensor.placeholder(shape=(x, y, z, 9), dtype="float32")


def _test_scalar_add(func):
    # pylint: disable=invalid-name
    x = tvm.nd.array(np.array(1.0).astype("float32"))
    y = tvm.nd.array(np.array(3.0).astype("float32"))
    z = func(x, y).numpy()
    # pylint: enable=invalid-name
    assert z.ndim == 0
    assert z.dtype == "float32"
    assert float(z) == 4.0


def _test_infer_sym(func, x, y, z):  # pylint: disable=invalid-name
    # pylint: disable=invalid-name
    a = tvm.nd.array(np.random.uniform(size=(x, y, 1)).astype("float32"))
    b = tvm.nd.array(np.random.uniform(size=(y, z, 5)).astype("float32"))
    c = func(a, b).numpy()
    # pylint: enable=invalid-name
    assert c.shape == (x, y, z, 9)


def _check_ir_equality(mod):
    # pylint: disable=import-outside-toplevel
    from tvm.script import ir as I
    from tvm.script import relax as R
    from tvm.script import tir as T

    # pylint: enable=import-outside-toplevel

    @I.ir_module
    class ExpectedModule:
        @R.function
        def scalar_add(
            a: R.Tensor((), dtype="float32"), b: R.Tensor((), dtype="float32")
        ) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                ext_scalar_add = R.call_dps_packed(
                    "ext_scalar_add", (a, b), out_sinfo=R.Tensor((), dtype="float32")
                )
                gv: R.Tensor((), dtype="float32") = ext_scalar_add
                R.output(gv)
            return gv

        @R.function
        def test_sym(
            a: R.Tensor(("x", "y", 1), dtype="float32"), b: R.Tensor(("y", "z", 5), dtype="float32")
        ) -> R.Tensor(("x", "y", "z", 9), dtype="float32"):
            x = T.int64()
            y = T.int64()
            z = T.int64()
            R.func_attr({"num_input": 2})
            with R.dataflow():
                ext_test_sym = R.call_dps_packed(
                    "ext_test_sym", (a, b), out_sinfo=R.Tensor((x, y, z, 9), dtype="float32")
                )
                gv1: R.Tensor((x, y, z, 9), dtype="float32") = ext_test_sym
                R.output(gv1)
            return gv1

    tvm.ir.assert_structural_equal(ExpectedModule, mod)


def _compile_cc(src: Path, dst: Path):
    # pylint: disable=import-outside-toplevel
    from tvm._ffi.base import py_str
    from tvm._ffi.libinfo import find_include_path

    # pylint: enable=import-outside-toplevel

    cmd = ["g++", str(src)]
    for include_path in find_include_path():
        cmd += ["-I", include_path]
    cmd += [
        "-DDMLC_USE_FOPEN64=0",
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
        "-c",
        "-fPIC",
        "-o",
        str(dst),
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "Compilation error:\n"
            msg += py_str(out)
            msg += "\nCommand line: " + " ".join(cmd)
            raise RuntimeError(msg)


def test_extern_object():
    with tempfile.TemporaryDirectory() as temp_dir_str:
        path = Path(temp_dir_str) / "main.o"
        _compile_cc(
            src=Path(__file__).parent / "frontend_nn_extern_module.cc",
            dst=path,
        )

        class TestModule(nn.Module):
            def __init__(self):
                self.ext_mod = None

            def _get_ext_mod(self):
                if self.ext_mod is None:
                    self.ext_mod = nn.ObjectModule(
                        {
                            "ext_scalar_add": _infer_scalar_add,
                            "ext_test_sym": _infer_test_sym,
                        },
                        path,
                    )
                    nn.add_extern(self.ext_mod)
                return self.ext_mod

            def scalar_add(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
                return self._get_ext_mod()["ext_scalar_add"](a, b)

            def test_sym(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
                return self._get_ext_mod()["ext_test_sym"](a, b)

        mod, _, ext_mods = TestModule().export_tvm(
            spec={
                "scalar_add": {
                    "a": spec.Tensor((), "float32"),
                    "b": spec.Tensor((), "float32"),
                },
                "test_sym": {
                    "a": spec.Tensor(("x", "y", 1), "float32"),
                    "b": spec.Tensor(("y", "z", 5), "float32"),
                },
            },
            allow_extern=True,
        )
        _check_ir_equality(mod)
        mod = AttachExternModules(ext_mods)(mod)  # pylint: disable=not-callable
        compiled = tvm.runtime.relax_vm.VirtualMachine(
            relax.build(mod, target="llvm"),
            device=tvm.cpu(),
        )
        _test_scalar_add(compiled["scalar_add"])
        _test_infer_sym(compiled["test_sym"], x=3, y=4, z=2)


def test_extern_source():
    source = Path(__file__).parent / "frontend_nn_extern_module.cc"

    class TestModule(nn.Module):
        def __init__(self):
            self.ext_mod = None

        def _get_ext_mod(self):
            if self.ext_mod is None:
                self.ext_mod = nn.SourceModule(
                    {
                        "ext_scalar_add": _infer_scalar_add,
                        "ext_test_sym": _infer_test_sym,
                    },
                    source_code=source,
                    source_format="cpp",
                )
                nn.add_extern(self.ext_mod)
            return self.ext_mod

        def scalar_add(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self._get_ext_mod()["ext_scalar_add"](a, b)

        def test_sym(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self._get_ext_mod()["ext_test_sym"](a, b)

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "scalar_add": {
                "a": spec.Tensor((), "float32"),
                "b": spec.Tensor((), "float32"),
            },
            "test_sym": {
                "a": spec.Tensor(("x", "y", 1), "float32"),
                "b": spec.Tensor(("y", "z", 5), "float32"),
            },
        },
        allow_extern=True,
    )
    _check_ir_equality(mod)
    mod = AttachExternModules(ext_mods)(mod)  # pylint: disable=not-callable
    compiled = tvm.runtime.relax_vm.VirtualMachine(
        relax.build(mod, target="llvm"),
        device=tvm.cpu(),
    )
    _test_scalar_add(compiled["scalar_add"])
    _test_infer_sym(compiled["test_sym"], x=3, y=4, z=2)


if __name__ == "__main__":
    test_extern_object()
    test_extern_source()
