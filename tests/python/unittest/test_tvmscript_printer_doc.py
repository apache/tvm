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
import pytest

from tvm.script.printer.doc import (
    LiteralDoc,
    IdDoc,
    AttrAccessDoc,
    IndexDoc,
    CallDoc,
    OperationKind,
    OperationDoc,
    LambdaDoc,
    TupleDoc,
    ListDoc,
    DictDoc,
    SliceDoc,
)


@pytest.mark.parametrize(
    "value",
    [None, "test", 0, 1, -2, 0.0, 1.5, -1.3, True, False],
)
def test_literal_doc_construction(value):
    doc = LiteralDoc(value)

    if isinstance(value, float):
        # FloatImm cannot be compared with Python's float directly
        assert float(doc.value) == pytest.approx(value)
    else:
        assert doc.value == value


def test_id_doc():
    doc = IdDoc("name")

    assert doc.name == "name"


def test_attr_access_doc():
    target = IdDoc("x")

    doc = AttrAccessDoc(target, "attribute")

    assert doc.value == target
    assert doc.name == "attribute"


@pytest.mark.parametrize(
    "indices",
    [
        [],
        [LiteralDoc(1)],
        [LiteralDoc(2), IdDoc("x")],
        [SliceDoc(LiteralDoc(1), LiteralDoc(2))],
        [SliceDoc(LiteralDoc(1)), IdDoc("y")],
    ],
)
def test_index_doc(indices):
    target = IdDoc("x")

    doc = IndexDoc(target, indices)

    assert doc.value == target
    assert list(doc.indices) == indices


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([], {}),
        ([LiteralDoc("arg")], {}),
        ([LiteralDoc("arg"), IdDoc("x")], {}),
        ([], {"x": LiteralDoc("x")}),
        ([], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg"), IdDoc("x")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
    ],
)
def test_call_doc(args, kwargs):
    target = IdDoc("x")

    doc = CallDoc(target, *args, **kwargs)

    assert doc.callee == target
    assert list(doc.args) == args
    assert dict(zip(doc.kwargs_keys, doc.kwargs_values)) == kwargs


@pytest.mark.parametrize(
    "operands",
    [
        [],
        [LiteralDoc(1)],
        [LiteralDoc(2), IdDoc("x")],
        [LiteralDoc(2), IdDoc("x"), LiteralDoc("y")],
    ],
)
def test_operation_doc(operands):
    # Here we just test the contructor and attr visitor of OperationDoc
    # so the choice of OperationKind doesn't matter
    operator = OperationKind.Add

    doc = OperationDoc(OperationKind.Add, operands)

    assert doc.kind == operator
    assert list(doc.operands) == operands


@pytest.mark.parametrize(
    "args",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_lambda_doc(args):
    body = LiteralDoc(1)

    doc = LambdaDoc(args, body)

    assert doc.body == body
    assert list(doc.args) == args


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_tuple_doc(elements):
    doc = TupleDoc(elements)

    assert list(doc.elements) == elements


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_list_doc(elements):
    doc = ListDoc(elements)

    assert list(doc.elements) == elements


@pytest.mark.parametrize(
    "content",
    [
        {},
        {LiteralDoc("k"): IdDoc("v")},
        {LiteralDoc("k"): IdDoc("v"), LiteralDoc("k2"): IdDoc("v2")},
    ],
)
def test_dict_doc(content):
    doc = DictDoc(content)

    assert dict(zip(doc.keys, doc.values)) == content


@pytest.mark.parametrize("start", [LiteralDoc(1), None])
@pytest.mark.parametrize("stop", [LiteralDoc(2), None])
@pytest.mark.parametrize("step", [LiteralDoc(3), None])
def test_slice_doc(start, stop, step):
    doc = SliceDoc(start, stop)

    assert doc.start == start
    assert doc.stop == stop


def test_expr_doc_attr_access():
    target = IdDoc("x")
    attr = "test"

    doc = target.attr(attr)

    assert doc.value == target
    assert doc.name == attr


@pytest.mark.parametrize(
    "indices",
    [
        (),
        LiteralDoc(1),
        SliceDoc(LiteralDoc(1), LiteralDoc(2)),
        (LiteralDoc(1),),
        (LiteralDoc(2), IdDoc("x")),
        (SliceDoc(LiteralDoc(1), LiteralDoc(2)),),
        (SliceDoc(LiteralDoc(1)), IdDoc("y")),
    ],
)
def test_expr_doc_get_item(indices):
    target = IdDoc("x")

    doc = target[indices]

    assert doc.value == target
    if not isinstance(indices, tuple):
        indices = (indices,)
    assert tuple(doc.indices) == indices


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([], {}),
        ([LiteralDoc("arg")], {}),
        ([LiteralDoc("arg"), IdDoc("x")], {}),
        ([], {"x": LiteralDoc("x")}),
        ([], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg"), IdDoc("x")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
    ],
)
def test_expr_doc_call_with(args, kwargs):
    target = IdDoc("x")

    doc = target.call(*args, **kwargs)

    assert doc.callee == target
    assert list(doc.args) == args
    assert dict(zip(doc.kwargs_keys, doc.kwargs_values)) == kwargs
