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

from typing import Optional

import pytest
from tvm.runtime import ObjectPath
from tvm.script import tir as T


@T.prim_func
def _func():
    T.evaluate(-1)
    T.evaluate(1)
    T.evaluate(2)
    T.evaluate(3)
    T.evaluate(4)
    T.evaluate(5)
    T.evaluate(6)
    T.evaluate(7)


def test_annotation_multi_object_paths():
    result = _func.with_attr("global_symbol", "main").script(
        path_to_annotate={
            ObjectPath.root().attr("body").attr("seq").array_index(1): "annotation 1",
            ObjectPath.root().attr("body").attr("seq").array_index(3): "annotation 3",
            ObjectPath.root().attr("body").attr("seq").array_index(5): "annotation 5",
            ObjectPath.root().attr("body").attr("seq").array_index(7): "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm.script import tir as T

@T.prim_func
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_annotate_from_multi_obj():
    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.seq[1]: "annotation 1",
            _func.body.seq[3]: "annotation 3",
            _func.body.seq[5]: "annotation 5",
            _func.body.seq[7]: "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm.script import tir as T

@T.prim_func
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_disable_concise_scoping_when_scope_annotated():
    @T.prim_func
    def _func():
        x = 1
        y = x + 1
        T.evaluate(y - 1)

    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.body: "annotation 1",
        }
    )
    assert (
        result
        == """# from tvm.script import tir as T

@T.prim_func
def main():
    x: T.int32 = 1
    # annotation 1
    with T.LetStmt(x + 1) as y:
        T.evaluate(y - 1)"""
    )
