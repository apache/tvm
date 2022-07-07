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

from tvm.tir import IntImm
from tvm.script.printer.doc import LiteralDoc


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
