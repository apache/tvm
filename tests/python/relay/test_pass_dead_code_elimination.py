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
from tvm import te
from tvm import relay
from tvm.relay import Function, transform
from tvm.relay.analysis import free_vars
from tvm.relay.op import log, add, equal, subtract
from tvm.relay.testing import inception_v3

import pytest


def optimize_source(source, passes):
    if not isinstance(passes, list):
        passes = [passes]

    optimize = tvm.transform.Sequential(passes)
    module = tvm.parser.parse(source)
    return optimize(module)


def optimize_and_check(before_source, after_source, passes):
    optimize_module = optimize_source(before_source, passes)
    after_module = tvm.parser.parse(after_source)
    print(optimize_module)
    print(after_module)
    assert tvm.ir.structural_equal(after_module, optimize_module)


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


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
