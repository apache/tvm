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

import numpy as np
import tvm
import pickle
from tvm import te
from tvm import nd, relay
from tvm.runtime import container as _container


def test_adt_constructor():
    arr = nd.array([1, 2, 3])
    fields = [arr, arr]
    y = _container.ADT(0, [arr, arr])

    assert len(y) == 2
    assert isinstance(y, _container.ADT)
    y[0:1][-1] == arr
    assert y.tag == 0
    assert isinstance(arr, nd.NDArray)


def test_tuple_object():
    x = relay.var(
        'x',
        type_annotation=relay.ty.TupleType([
            relay.ty.TensorType((), 'int32'),
            relay.ty.TensorType((), 'int32')
        ]))

    fn = relay.Function([x], relay.expr.TupleGetItem(x, 0))
    mod = tvm.IRModule.from_expr(fn)

    exe = relay.create_executor(
        kind="vm", mod=mod, ctx=nd.cpu(), target="llvm")
    f = exe.evaluate()
    value_tuple = _container.tuple_object(
        [nd.array(np.array(11)),
         nd.array(np.array(12))])
    # pass an ADT object to evaluate
    out = f(value_tuple)
    tvm.testing.assert_allclose(out.asnumpy(), np.array(11))


def test_string():
    s = tvm.runtime.String("xyz")

    assert isinstance(s, tvm.runtime.String)
    assert isinstance(s, str)
    assert s.startswith("xy")
    assert s + "1" == "xyz1"
    y = tvm.testing.echo(s)
    assert isinstance(y, tvm.runtime.String)
    assert s.__tvm_object__.same_as(y.__tvm_object__)
    assert s == y

    x = tvm.ir.load_json(tvm.ir.save_json(y))
    assert isinstance(x, tvm.runtime.String)
    assert x == y

    # test pickle
    z = pickle.loads(pickle.dumps(s))
    assert isinstance(z, tvm.runtime.String)
    assert s == z


if __name__ == "__main__":
    test_string()
    test_adt_constructor()
    test_tuple_object()
