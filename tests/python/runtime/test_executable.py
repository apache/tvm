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
"""Tests for the Executable class."""

import os
import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm.runtime import Executable
from tvm.script import tir as T


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def add(
        A: T.Buffer((10,), "float32"),
        B: T.Buffer((10,), "float32"),
        C: T.Buffer((10,), "float32"),
    ):
        for i in range(10):
            C[i] = A[i] + B[i]


def test_executable_init():
    """Test initialization of Executable class."""
    lib = tvm.tir.build(MyModule, target="llvm")
    executable = Executable(lib)

    assert executable.mod is lib
    assert executable._jitted_mod is None


def test_executable_getitem():
    """Test __getitem__ method of Executable class."""
    lib = tvm.tir.build(MyModule, target="llvm")
    executable = Executable(lib)

    # Jit the module first
    executable.jit()

    # Test __getitem__
    add_func = executable["add"]

    # Verify the function works
    a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
    b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
    c = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

    add_func(a, b, c)

    # Check results
    tvm.testing.assert_allclose(c.numpy(), np.array([3.0] * 10, dtype="float32"))


def test_executable_jit_already_jitted():
    """Test jit method when module is already jitted."""
    lib = tvm.tir.build(MyModule, target="llvm")
    executable = Executable(lib)

    # First jit call
    jitted_mod1 = executable.jit()

    # Second jit call should return the cached jitted module
    jitted_mod2 = executable.jit()
    assert jitted_mod2 is jitted_mod1

    # Test with force_recompile
    jitted_mod3 = executable.jit(force_recompile=True)
    # The module might be different after force recompilation

    # Verify both modules work correctly
    a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
    b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
    c1 = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))
    c2 = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

    jitted_mod1["add"](a, b, c1)
    jitted_mod3["add"](a, b, c2)

    tvm.testing.assert_allclose(c1.numpy(), np.array([3.0] * 10, dtype="float32"))
    tvm.testing.assert_allclose(c2.numpy(), np.array([3.0] * 10, dtype="float32"))


def test_executable_export_library():
    """Test export_library method."""
    lib = tvm.tir.build(MyModule, target="llvm")
    executable = Executable(lib)

    # Create a temporary directory for the library
    temp_dir = tempfile.mkdtemp()
    try:
        lib_path = os.path.join(temp_dir, "test_lib.so")
        executable.export_library(lib_path)

        # Verify the library was created
        assert os.path.exists(lib_path)

        # Load the library back
        loaded_mod = tvm.runtime.load_module(lib_path)
        assert loaded_mod is not None

        # Test the loaded module
        a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
        b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
        c = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

        loaded_mod["add"](a, b, c)

        # Check results
        tvm.testing.assert_allclose(c.numpy(), np.array([3.0] * 10, dtype="float32"))
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)


def test_executable_export_library_with_workspace():
    """Test export_library method with workspace_dir."""
    lib = tvm.tir.build(MyModule, target="llvm")
    executable = Executable(lib)

    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    workspace_dir = tempfile.mkdtemp()

    try:
        lib_path = os.path.join(temp_dir, "test_lib.so")
        executable.export_library(lib_path, workspace_dir=workspace_dir)

        # Verify the library was created
        assert os.path.exists(lib_path)

        # Load the library back
        loaded_mod = tvm.runtime.load_module(lib_path)
        assert loaded_mod is not None

        # Test the loaded module
        a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
        b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
        c = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

        loaded_mod["add"](a, b, c)

        # Check results
        tvm.testing.assert_allclose(c.numpy(), np.array([3.0] * 10, dtype="float32"))
    finally:
        # Clean up
        for directory in [temp_dir, workspace_dir]:
            if os.path.exists(directory):
                import shutil

                shutil.rmtree(directory)


def test_executable_integration():
    """Integration test for Executable with a simple TVM module."""
    # Create target and build
    target = tvm.target.Target("llvm")
    lib = tvm.tir.build(MyModule, target=target)

    # Create an executable
    executable = Executable(lib)

    # Test jit
    jitted_mod = executable.jit()
    assert jitted_mod is not None

    # Test __getitem__
    add_func = executable["add"]
    assert add_func is not None

    # Test the function works
    a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
    b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
    c = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

    add_func(a, b, c)

    # Check results
    tvm.testing.assert_allclose(c.numpy(), np.array([3.0] * 10, dtype="float32"))

    # Test export_library
    temp_dir = tempfile.mkdtemp()
    try:
        lib_path = os.path.join(temp_dir, "test_lib.so")
        executable.export_library(lib_path)

        # Verify the library was created
        assert os.path.exists(lib_path)

        # Load the library back
        loaded_mod = tvm.runtime.load_module(lib_path)
        assert loaded_mod is not None

        # Test the loaded module
        loaded_add = loaded_mod["add"]
        c_loaded = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))
        loaded_add(a, b, c_loaded)

        # Check results
        tvm.testing.assert_allclose(c_loaded.numpy(), np.array([3.0] * 10, dtype="float32"))

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)


def test_executable_jit_force_recompile():
    """Test jit method with force_recompile=True."""
    # Create target and build
    target = tvm.target.Target("c")
    lib = tvm.tir.build(MyModule, target=target)

    # Create an executable
    executable = Executable(lib)

    # First jit call
    jitted_mod1 = executable.jit()

    # Second jit call without force_recompile should return the same module
    jitted_mod2 = executable.jit()
    assert jitted_mod1 is jitted_mod2

    # Third jit call with force_recompile should return a new module
    jitted_mod3 = executable.jit(force_recompile=True)
    assert jitted_mod3 is not jitted_mod1

    # Test the function works
    a = tvm.nd.array(np.array([1.0] * 10, dtype="float32"))
    b = tvm.nd.array(np.array([2.0] * 10, dtype="float32"))
    c = tvm.nd.array(np.array([0.0] * 10, dtype="float32"))

    jitted_mod3["add"](a, b, c)

    # Check results
    tvm.testing.assert_allclose(c.numpy(), np.array([3.0] * 10, dtype="float32"))


if __name__ == "__main__":
    tvm.testing.main()
