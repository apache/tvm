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

import inspect

import tvm
import tvm.script
from tvm import tir


def loops() -> None:
    for i in tir.parallel(0, 2):
        for j in tir.serial(0, 1):
            for z in tir.vectorized(3, 4):
                tir.evaluate(0)


def test_loops():
    _, start_line = inspect.getsourcelines(loops)
    parsed = tvm.script.tir(loops)

    assert parsed.span.line == start_line

    assert parsed.body.span.line == start_line + 1
    assert parsed.body.min.span.column == 27
    assert parsed.body.extent.span.column == 30
    assert parsed.body.extent.span.line == start_line + 1

    assert parsed.body.body.span.line == start_line + 2
    assert parsed.body.body.loop_var.span.line == start_line + 2
    assert parsed.body.body.loop_var.span.column == 13

    assert parsed.body.body.body.span.line == start_line + 3
    assert parsed.body.body.body.span.column == 22

    assert parsed.body.body.body.body.span.line == start_line + 4
    assert parsed.body.body.body.body.span.column == 17


def statements() -> None:
    tir.evaluate(1)
    tir.evaluate("test")


def test_statements():
    _, start_line = inspect.getsourcelines(statements)
    parsed = tvm.script.tir(statements)

    assert parsed.body.span.line == start_line + 1

    assert parsed.body[0].span.line == start_line + 1
    assert parsed.body[0].span.column == 5

    assert parsed.body[0].span.line == start_line + 1
    assert parsed.body[0].span.column == 5


if __name__ == "__main__":
    test_loops()
    test_statements()
