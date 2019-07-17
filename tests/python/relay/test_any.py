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
from tvm import relay
from tvm.relay import Kind, transform
from tvm.relay.loops import while_loop
from tvm.relay.testing import run_infer_type as infer_type

def int32(val):
    return relay.const(val, 'int32')

def test_arange_with_dynamic_shape():
    m, n, k = relay.ShapeVar('m'), relay.ShapeVar('n'), relay.ShapeVar('k')
    x = relay.var('x', shape=(m.var, n.var, k.var), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, 'int32'))
    y2 = relay.op.arange(y1)
    ex = relay.create_executor()
    f = relay.Function([x], y2, type_params=[m, n, k])
    # TODO(@jroesch): Restore after code generation.
    # data = np.random.rand(10, 5, 3).astype('float32')
    # result = ex.evaluate(f)(data)
    # np.testing.assert_allclose(result.asnumpy(), np.array(range(10)))

def test_dynamic_concat():
    """
    fn @concat_loop(%i: int32, %st: (any, 1)) -> (any, 1) {
        if (%i < 10) {
            let %i = reshape(cast(i, "float32"), newshape=(1, ))
            let %new_st = concatenate((st, i), axis=0)
            concat_loop(%i + 1, )
        } else {
            st
        }
    }
    """
    # Initial Values.
    i = relay.var('i', shape=(), dtype='int32')
    st = relay.var('st', shape=(relay.Any(), 1), dtype='int32')

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1,1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var('start', shape=(), dtype='int32')
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    func = infer_type(func)
    # TODO(@jroesch, @haichen): We should restore this code when codegeneration
    # is merged
    # ret_shape = func.checked_type.ret_type.shape
    # assert len(ret_shape) == 2, "expected 2-dim output"
    # assert relay.ir_pass.alpha_eq(ret_shape[0], relay.Any())
    # import pdb; pdb.set_trace()
    # mod = relay.module.Module()
    # print(relay.ir_pass.infer_type(func, mod=mod))
    # ret = relay.Call(loop, [relay.const(0, 'int32'), init])
    # mod[mod.entry_func] = relay.Function([], ret)
    # print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))

    # initial = np.array(0.0, dtype='float32').reshape((1,))
    # iter_stop = np.array(10, dtype='int32')
    # ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
    # result = ex.evaluate(mod.entry_func)()
    # np.testing.assert_allclose(result.asnumpy(), np.array(range(10)))

def test_dynamic_concat_with_wrong_annotation():
    """
    v0.0.1
    fn (%start: int32) {
        %7 = {
            let %while_loop = fn (%i: int32, %st: Tensor[(1, 1), int32]) {
            %0 = less(%i, 10)
            %1 = min(%0)
            if (%1) {
                %2 = add(%i, 1)
                %3 = reshape(%i, newshape=[1, 1])
                %4 = (%st, %3)
                /* The result of concat should be 1,1 but it is 2, 1. */
                %5 = concatenate(%4)
                %while_loop(%2, %5)
            } else {
                (%i, %st)
            }
        }
        %6 = reshape(0, newshape=[1, 1])
        %while_loop(%start, %6)
    }
    %7.1
    }
    """
    # Initial Values.
    i = relay.var('i', shape=(), dtype='int32')
    st = relay.var('st', shape=(1, 1), dtype='int32')

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1,1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var('start', shape=(), dtype='int32')
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    try:
        func = infer_type(func)
        assert False
    except Exception as e:
        assert "in particular dimension 0 conflicts 2 does not match 1" in str(e)

if __name__ == "__main__":
    test_arange_with_dynamic_shape()
    test_dynamic_concat()
    test_dynamic_concat_with_wrong_annotation()
