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

import tvm
import tvm.testing

from tvm.contrib import utils


@tvm.testing.requires_llvm
def test_import_static_library():
    from tvm import te

    # Generate two LLVM modules.
    A = te.placeholder((1024,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    irmod0 = tvm.IRModule.from_expr(
        te.create_prim_func([A, B]).with_attr("global_symbol", "myadd0")
    )
    irmod1 = tvm.IRModule.from_expr(
        te.create_prim_func([A, B]).with_attr("global_symbol", "myadd1")
    )

    mod0 = tvm.tir.build(irmod0, target="llvm")
    mod1 = tvm.tir.build(irmod1, target="llvm")

    assert mod0.implements_function("myadd0")
    assert mod1.implements_function("myadd1")
    assert mod1.is_dso_exportable

    # mod1 is currently an 'llvm' module.
    # Save and reload it as a vanilla 'static_library'.
    temp = utils.tempdir()
    mod1_o_path = temp.relpath("mod1.o")
    mod1.save(mod1_o_path)
    mod1_o = tvm.runtime.load_static_library(mod1_o_path, ["myadd1"])
    assert mod1_o.implements_function("myadd1")
    assert mod1_o.is_dso_exportable

    # Import mod1 as a static library into mod0 and compile to its own DSO.
    mod0.import_module(mod1_o)
    mod0_dso_path = temp.relpath("mod0.so")
    mod0.export_library(mod0_dso_path)

    # The imported mod1 is statically linked into mod0.
    loaded_lib = tvm.runtime.load_module(mod0_dso_path)
    assert loaded_lib.type_key == "library"
    assert len(loaded_lib.imported_modules) == 0
    assert loaded_lib.implements_function("myadd0")
    assert loaded_lib.get_function("myadd0")
    assert loaded_lib.implements_function("myadd1")
    assert loaded_lib.get_function("myadd1")
    assert not loaded_lib.is_dso_exportable


if __name__ == "__main__":
    tvm.testing.main()
