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

import pickle
from tvm import ffi as tvm_ffi


def test_string():
    fecho = tvm_ffi.get_global_func("testing.echo")
    s = tvm_ffi.String("hello")
    assert isinstance(s, tvm_ffi.String)
    s2 = fecho(s)
    assert s2.__tvm_ffi_object__.same_as(s.__tvm_ffi_object__)

    s3 = tvm_ffi.convert("hello")
    assert isinstance(s3, tvm_ffi.String)
    assert isinstance(s3, str)

    s4 = pickle.loads(pickle.dumps(s))
    assert s4 == "hello"
    assert isinstance(s4, tvm_ffi.String)


def test_bytes():
    fecho = tvm_ffi.get_global_func("testing.echo")
    b = tvm_ffi.Bytes(b"hello")
    assert isinstance(b, tvm_ffi.Bytes)
    b2 = fecho(b)
    assert b2.__tvm_ffi_object__.same_as(b.__tvm_ffi_object__)

    b3 = tvm_ffi.convert(b"hello")
    assert isinstance(b3, tvm_ffi.Bytes)
    assert isinstance(b3, bytes)

    b4 = tvm_ffi.convert(bytearray(b"hello"))
    assert isinstance(b4, tvm_ffi.Bytes)
    assert isinstance(b4, bytes)

    b5 = pickle.loads(pickle.dumps(b))
    assert b5 == b"hello"
    assert isinstance(b5, tvm_ffi.Bytes)
