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
import platform
from tvm import ffi as tvm_ffi


def test_parse_traceback():
    traceback = """
    File "test.py", line 1, in <module>
    File "test.py", line 3, in run_test
    """
    parsed = tvm_ffi.error._parse_traceback(traceback)
    assert len(parsed) == 2
    assert parsed[0] == ("test.py", 1, "<module>")
    assert parsed[1] == ("test.py", 3, "run_test")


def test_error_from_cxx():
    test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")

    try:
        test_raise_error("ValueError", "error XYZ")
    except ValueError as e:
        assert e.__tvm_ffi_error__.kind == "ValueError"
        assert e.__tvm_ffi_error__.message == "error XYZ"
        assert e.__tvm_ffi_error__.traceback.find("TestRaiseError") != -1

    fapply = tvm_ffi.convert(lambda f, *args: f(*args))

    with pytest.raises(TypeError):
        fapply(test_raise_error, "TypeError", "error XYZ")

    # wrong number of arguments
    with pytest.raises(TypeError):
        tvm_ffi.convert(lambda x: x)()


@pytest.mark.skipif(
    "32bit" in platform.architecture(),
    reason="libbacktrace file name support is not available in i386 yet",
)
def test_error_from_nested_pyfunc():
    fapply = tvm_ffi.convert(lambda f, *args: f(*args))
    cxx_test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")
    cxx_test_apply = tvm_ffi.get_global_func("testing.apply")

    record_object = []

    def raise_error():
        try:
            fapply(cxx_test_raise_error, "ValueError", "error XYZ")
        except ValueError as e:
            assert e.__tvm_ffi_error__.kind == "ValueError"
            assert e.__tvm_ffi_error__.message == "error XYZ"
            assert e.__tvm_ffi_error__.traceback.find("TestRaiseError") != -1
            record_object.append(e.__tvm_ffi_error__)
            raise e

    try:
        cxx_test_apply(raise_error)
    except ValueError as e:
        traceback = e.__tvm_ffi_error__.traceback
        assert e.__tvm_ffi_error__.same_as(record_object[0])
        assert traceback.count("TestRaiseError") == 1
        assert traceback.count("TestApply") == 1
        assert traceback.count("<lambda>") == 1
        pos_cxx_raise = traceback.find("TestRaiseError")
        pos_cxx_apply = traceback.find("TestApply")
        pos_lambda = traceback.find("<lambda>")
        assert pos_cxx_raise > pos_lambda
        assert pos_lambda > pos_cxx_apply


def test_error_traceback_update():
    fecho = tvm_ffi.get_global_func("testing.echo")

    def raise_error():
        raise ValueError("error XYZ")

    try:
        raise_error()
    except ValueError as e:
        ffi_error = tvm_ffi.convert(e)
        assert ffi_error.traceback.find("raise_error") != -1

    def raise_cxx_error():
        cxx_test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")
        cxx_test_raise_error("ValueError", "error XYZ")

    try:
        raise_cxx_error()
    except ValueError as e:
        assert e.__tvm_ffi_error__.traceback.find("raise_cxx_error") == -1
        ffi_error1 = tvm_ffi.convert(e)
        ffi_error2 = fecho(e)
        assert ffi_error1.traceback.find("raise_cxx_error") != -1
        assert ffi_error2.traceback.find("raise_cxx_error") != -1
