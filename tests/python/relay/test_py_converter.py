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
from tvm import relay
from tvm.relay.testing import to_python, run_as_python
from tvm.relay.backend.interpreter import TensorValue, TupleValue, RefValue

def test_create_empty_tuple():
    empty = relay.Tuple([])
    tup_val = run_as_python(empty)
    assert isinstance(tup_val, TupleValue)
    assert len(tup_val.fields) == 0


def test_create_scalar():
    scalar = relay.const(1)
    tensor_val = run_as_python(scalar)
    assert isinstance(tensor_val, TensorValue)
    assert tensor_val.data.asnumpy() == 1


def test_create_nested_tuple():
    relay_tup = relay.Tuple([
        relay.const(1), relay.const(2),
        relay.Tuple([
            relay.const(3),
            relay.const(4)
        ])
    ])
    tup_val = run_as_python(relay_tup)
    assert isinstance(tup_val, TupleValue)
    assert len(tup_val.fields) == 3
    for i in range(2):
        assert isinstance(tup_val.fields[i], TensorValue)
        assert tup_val.fields[i].data.asnumpy() == i + 1
    assert isinstance(tup_val.fields[2], TupleValue)
    for i in range(2):
        assert isinstance(tup_val.fields[2].fields[i], TensorValue)
        assert tup_val.fields[2].fields[i].data.asnumpy() == i + 3


def test_create_let():
    v = relay.Var('v')
    let = relay.Let(v, relay.Tuple([]), relay.Tuple([v, v]))
    tup_val = run_as_python(let)
    assert isinstance(tup_val, TupleValue)
    assert len(tup_val.fields) == 2
    assert isinstance(tup_val.fields[0], TupleValue)
    assert len(tup_val.fields[0].fields) == 0
    assert isinstance(tup_val.fields[1], TupleValue)
    assert len(tup_val.fields[1].fields) == 0


def test_create_ref():
    relay_ref = relay.RefCreate(relay.Tuple([]))
    ref_val = run_as_python(relay_ref)
    assert isinstance(ref_val, RefValue)
    assert isinstance(ref_val.value, TupleValue)
    assert len(ref_val.value.fields) == 0


def test_ref_read():
    v = relay.Var('v')
    assign = relay.Let(v, relay.RefCreate(relay.Tuple([])), relay.RefRead(v))
    read_val = run_as_python(assign)
    assert isinstance(read_val, TupleValue)
    assert len(read_val.fields) == 0


def test_ref_write():
    # check that the result of a ref write is an empty tuple
    v = relay.Var('v')
    initial_write = relay.Let(v, relay.RefCreate(relay.Tuple([relay.const(1)])),
                              relay.RefWrite(v, relay.Tuple([relay.const(2)])))
    write_val = run_as_python(initial_write)
    assert isinstance(write_val, TupleValue)
    assert len(write_val.fields) == 0

    # now ensure that the value, once written, can be read back
    read_after_write = relay.Let(v, relay.RefCreate(relay.Tuple([relay.const(1)])),
                                 relay.Let(relay.Var('_'),
                                           relay.RefWrite(v, relay.Tuple([relay.const(2)])),
                                           relay.RefRead(v)))
    read_val = run_as_python(read_after_write)
    assert isinstance(read_val, TupleValue)
    assert len(read_val.fields) == 1
    assert isinstance(read_val.fields[0], TensorValue)
    assert read_val.fields[0].asnumpy() == 2
