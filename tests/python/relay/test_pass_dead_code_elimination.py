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
from tvm import relay
from tvm.relay import Function, transform
from tvm.relay.testing import inception_v3
import numpy as np
import pytest

cpu_scope = tvm.target.VirtualDevice(tvm.cpu(), tvm.target.Target("llvm"))
metatable = {"VirtualDevice": [cpu_scope]}
core = tvm.IRModule()
core.import_from_std("core.rly")


def optimize_and_check(before_program, after_program, passes):
    if isinstance(before_program, str):
        before_program = tvm.relay.parse(before_program)
    if isinstance(after_program, str):
        after_program = tvm.relay.parse(after_program)
    if not isinstance(passes, list):
        passes = [passes]
    optimize = tvm.transform.Sequential(passes)
    optimized_program = optimize(before_program)
    print("Actual:")
    print(optimized_program)
    print("Expected:")
    print(after_program)
    assert tvm.ir.structural_equal(optimized_program, after_program, map_free_vars=True)


def test_dead_let():
    before_program = """
    #[version = "0.0.5"]
    def @main(%z: int) {
        let %x = 1;
        %z
    }
    """
    after_program = """
    #[version = "0.0.5"]
    def @main(%z: int) {
        %z
    }
    """
    optimize_and_check(before_program, after_program, transform.DeadCodeElimination())


def test_one_live_let():
    before_program = """
    #[version = "0.0.5"]
    def @main(%z: int) {
        let %x = 1;
        let %y = 2;
        %x + %x
    }
    """
    after_program = """
    #[version = "0.0.5"]
    def @main(%z: int) {
        let %x = 1;
        %x + %x
    }
    """
    optimize_and_check(before_program, after_program, transform.DeadCodeElimination())


def test_nested_let():
    before_program = """
    #[version = "0.0.5"]
    def @main(%d: int, %b: int) {
        let %a = %b;
        let %c = %d;
        %c
    }
    """
    after_program = """
    #[version = "0.0.5"]
    def @main(%d: int, %b: int) {
        let %c = %d;
        %c
    }
    """
    optimize_and_check(before_program, after_program, transform.DeadCodeElimination())


def test_live_recursion():
    before_program = """
    #[version = "0.0.5"]
    def @main() {
        let %f = fn (%n: int, %data: int) -> int {
            if (%n == 0) {
                %data
            } else {
                %f(%n - 1, log(%data))
            }
        };
        %f(2, 10000)
    }
    """

    after_program = """
    #[version = "0.0.5"]
    def @main() {
        let %f = fn (%n: int, %data: int) -> int {
            if (%n == 0) {
                %data
            } else {
                %f(%n - 1, log(%data))
            }
        };
        %f(2, 10000)
    }
    """

    optimize_and_check(
        before_program, after_program, [transform.DeadCodeElimination(), transform.InferType()]
    )


def test_dead_recursion():
    before_program = """
    #[version = "0.0.5"]
    def @main() {
        let %f = fn (%n: int, %data: int) -> int {
            if (%n == 0) {
                %data
            } else {
                %f(%n - 1, log(%data))
            }
        };
        ()
    }
    """

    after_program = """
    #[version = "0.0.5"]
    def @main() {
        ()
    }
    """

    optimize_and_check(
        before_program, after_program, [transform.DeadCodeElimination(), transform.InferType()]
    )


def test_add_with_let():
    before_program = """
    #[version = "0.0.5"]
    def @main() {
        (let %a = 1; 3) + 2
    }
    """

    after_program = """
    #[version = "0.0.5"]
    def @main() {
        3 + 2
    }
    """

    optimize_and_check(
        before_program, after_program, [transform.DeadCodeElimination(), transform.InferType()]
    )


def test_tuple_get_item():
    before_program = """
    #[version = "0.0.5"]
    def @main() {
        let %a = 100;
        (1, 2, 3, 4).0
    }
    """

    after_program = """
    #[version = "0.0.5"]
    def @main() {
        (1, 2, 3, 4).0
    }
    """

    optimize_and_check(before_program, after_program, transform.DeadCodeElimination())


def test_inline_into_function():
    """Don't inline across function boundaries."""
    before_program = """
    #[version = "0.0.5"]
    def @main() {
        let %x = 1 + 1;
        let %f = fn (%y: int) -> int {
          let %z = %y + %y;
          %x + %z
        };
        (%f(2), %f(3))
    }
    """

    after_program = """
    #[version = "0.0.5"]
    def @main() {
        let %x = 1 + 1;
        let %f = fn (%y: int) -> int {
          %x + (%y + %y)
        };
        (%f(2), %f(3))
    }
    """

    optimize_and_check(
        before_program, after_program, transform.DeadCodeElimination(inline_once=True)
    )


def test_impure_op():
    shape = np.array([64, 2])
    metatable = {
        "VirtualDevice": [cpu_scope],
        "relay.Constant": [relay.const(shape, dtype="int64")],
    }
    """Don't elide calls to side-effecting operators."""
    before_program = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main() {
           let %size: int64 = cast(1024, dtype="int64");
           let %alignment: int64 = cast(64, dtype="int64");
           let %x = memory.alloc_storage(%size, meta[relay.Constant][0], %alignment, virtual_device=meta[VirtualDevice][0]);
           let %_ = memory.kill(%x);
           0
        }
        """,
        "from_string",
        core,
        metatable,
    )

    after_program = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main() {
           %0 = memory.alloc_storage(cast(1024, dtype="int64"),
                                     meta[relay.Constant][0],
                                     cast(64, dtype="int64"),
                                     virtual_device=meta[VirtualDevice][0]);
           let %_ = memory.kill(%0);
           0
        }
        """,
        "from_string",
        core,
        metatable,
    )

    optimize_and_check(
        before_program, after_program, transform.DeadCodeElimination(inline_once=True)
    )


def test_impure_func():
    shape = np.array([64, 2])
    metatable = {
        "VirtualDevice": [cpu_scope],
        "relay.Constant": [relay.const(shape, dtype="int64")],
    }
    """Don't elide calls to side-effecting functions."""
    before_program = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @f() -> int {
           let %size: int64 = cast(1024, dtype="int64");
           let %alignment: int64 = cast(64, dtype="int64");
           let %x = memory.alloc_storage(%size, meta[relay.Constant][0], %alignment, virtual_device=meta[VirtualDevice][0]);
           let %_ = memory.kill(%x);
           0
        }
        def @main() -> int {
           let %y = @f();
           0
        }
        """,
        "from_string",
        core,
        metatable,
    )

    after_program = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @f() -> int {
           %0 = memory.alloc_storage(cast(1024, dtype="int64"),
                                     meta[relay.Constant][0],
                                     cast(64, dtype="int64"),
                                     virtual_device=meta[VirtualDevice][0]);
           let %_ = memory.kill(%0);
           0
        }
        def @main() -> int {
            let %y = @f();
            0
        }
        """,
        "from_string",
        core,
        metatable,
    )

    optimize_and_check(
        before_program, after_program, transform.DeadCodeElimination(inline_once=True)
    )


def test_refs():
    """Don't elide expressions with reference create/read/write side effects"""
    before_program = """
    #[version = "0.0.5"]
    def @f(%r) -> int {
        let %v = ref_read(%r);
        let %u = ref_write(%r, %v + 1);
        %v
    }
    def @main() -> int {
        let %r = ref(0);
        let %y = @f(%r);
        let %z = @f(%r);
        %z
    }
    """

    after_program = before_program

    optimize_and_check(
        before_program,
        after_program,
        [transform.InferType(), transform.DeadCodeElimination(inline_once=True)],
    )


def test_complexity():
    mod = transform.InferType()(
        tvm.IRModule.from_expr(inception_v3.get_net(1, 1000, (3, 299, 299), "float32"))
    )

    optimize_and_check(mod, mod, transform.DeadCodeElimination(inline_once=True))


if __name__ == "__main__":
    tvm.testing.main()
