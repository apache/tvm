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

from tvm.runtime import ObjectPath
from tvm.script.printer.doc import IdDoc
from tvm.script.printer.frame import MetadataFrame, VarDefFrame
from tvm.script.printer.ir_docsifier import IRDocsifier
from tvm.tir import Var


@pytest.fixture
def ir_docsifier():
    """
    Creates an IRDocsifier instance with a special dispatch token.
    """
    _ir_docsifier = IRDocsifier({})
    with _ir_docsifier.dispatch_token(f"{__file__}"):
        yield _ir_docsifier


def _get_id_doc_printer(id_name):
    def printer(obj, object_path, ir_docsifier):  # pylint: disable=unused-argument
        return IdDoc(id_name)

    return printer


# Because the dispatch table is global, tests should only set dispatch function under
# unique dispatch token.
IRDocsifier.set_dispatch(Var, _get_id_doc_printer("x"), f"{__file__}")


def test_set_dispatch(ir_docsifier):
    IRDocsifier.set_dispatch(Var, _get_id_doc_printer("x2"), f"{__file__}-2")
    with ir_docsifier.dispatch_token(f"{__file__}-2"):
        doc = ir_docsifier.as_doc(Var("x", dtype="int8"), ObjectPath.root())
        assert doc.name == "x2"

    doc = ir_docsifier.as_doc(Var("x", dtype="int8"), ObjectPath.root())
    assert doc.name == "x"


def test_as_doc(ir_docsifier):
    object_path = ObjectPath.root()
    doc = ir_docsifier.as_doc(Var("x", "int8"), ObjectPath.root())
    assert doc.name == "x"
    assert list(doc.source_paths) == [object_path]


def test_with_dispatch_token(ir_docsifier):
    initial_token_count = len(ir_docsifier.dispatch_tokens)

    with ir_docsifier.dispatch_token("tir"):
        assert len(ir_docsifier.dispatch_tokens) == initial_token_count + 1

    assert len(ir_docsifier.dispatch_tokens) == initial_token_count


def test_with_frame(ir_docsifier):
    initial_frame_count = len(ir_docsifier.frames)

    frame = VarDefFrame()
    is_callback_called = False

    def callback():
        nonlocal is_callback_called
        is_callback_called = True

    frame.add_exit_callback(callback)

    with ir_docsifier.frame(frame):
        assert len(ir_docsifier.frames) == initial_frame_count + 1
        assert not is_callback_called

    assert len(ir_docsifier.frames) == initial_frame_count
    assert is_callback_called


def test_get_frame(ir_docsifier):
    with ir_docsifier.frame(VarDefFrame()) as frame_a:
        assert ir_docsifier.get_frame(MetadataFrame) is None
        assert ir_docsifier.get_frame(VarDefFrame) == frame_a

        with ir_docsifier.frame(VarDefFrame()) as frame_b:
            assert ir_docsifier.get_frame(MetadataFrame) is None
            assert ir_docsifier.get_frame(VarDefFrame) == frame_b

            with ir_docsifier.frame(MetadataFrame()) as frame_c:
                assert ir_docsifier.get_frame(MetadataFrame) == frame_c
                assert ir_docsifier.get_frame(VarDefFrame) == frame_b

            assert ir_docsifier.get_frame(MetadataFrame) is None
            assert ir_docsifier.get_frame(VarDefFrame) == frame_b

        assert ir_docsifier.get_frame(MetadataFrame) is None
        assert ir_docsifier.get_frame(VarDefFrame) == frame_a
