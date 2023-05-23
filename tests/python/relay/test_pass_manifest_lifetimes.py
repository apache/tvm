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
from tvm.relay import Function, transform
from tvm.relay.testing import inception_v3
import pytest
import sys


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


def test_simple_linear():
    before_program = """
    #[version = "0.0.5"]
    def @main(%x: int) {
        let %y = %x + %x;
        let %z = %y + %y;
        let %w = %z + %z;
        %w
    }
    """
    after_program = """
    #[version = "0.0.5"]
    def @main(%x: int) {
        let %y = %x + %x;
        let %_0 = memory.kill(%x);
        let %z = %y + %y;
        let %_1 = memory.kill(%y);
        let %w = %z + %z;
        let %_2 = memory.kill(%z);
        %w
    }
    """
    optimize_and_check(before_program, after_program, transform.ManifestLifetimes())


def test_simple_if():
    before_program = """
    #[version = "0.0.5"]
    def @main(%x: int) {
        let %y = cast(%x, dtype="bool");
        let %z = if (%y) {
            let %v0 = %x + %x;
            let %v1 = %v0 * 2;
            %v1
        } else {
            %x
        };
        %z
    }
    """
    after_program = """
    #[version = "0.0.5"]
    def @main(%x: int) {
        let %y = cast(%x, dtype="bool");
        let %z = if (%y) {
            let %v0 = %x + %x;
            let %_0 = memory.kill(%x);
            let %v1 = %v0 * 2;
            let %_1 = memory.kill(%v0);
            %v1
        } else {
            %x
        };
        let %_1 = memory.kill(%y);
        %z
    }
    """
    optimize_and_check(before_program, after_program, transform.ManifestLifetimes())


def test_simple_match():
    before_program = """
    #[version = "0.0.5"]
    type List[A] {
        Cons(A, List[A]),
        Nil,
    }
    def @main(%x: int) {
        let %l : List[int] = Nil;
        let %m = (match (%l) {
            Cons(%head, %rest) => {
                let %y = %x + 1;
                let %z = %y + %y;
                %z
            },
            Nil => -1,
        });
        %m
    }
    """
    after_program = """
    #[version = "0.0.5"]
    type List[A] {
        Cons(A, List[A]),
        Nil,
    }
    def @main(%x: int) {
        let %l : List[int] = Nil;
        let %m = (match (%l) {
            Cons(%head, %rest) => {
                let %y = %x + 1;
                let %_0 = memory.kill(%x);
                let %z = %y + %y;
                let %_1 = memory.kill(%y);
                /* TODO: %head and %rest should be immediately killed */
                %z
            },
            Nil => -1
        });
        let %_2 = memory.kill(%l);
        %m
    }
    """
    optimize_and_check(before_program, after_program, transform.ManifestLifetimes())


if __name__ == "__main__":
    tvm.testing.main()
