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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks

import numpy as np

import tvm
from tvm import relay

data0 = relay.var("data0", shape=(1, relay.Any()))
data1 = relay.var("data1", shape=(1, relay.Any()))

r0 = relay.cast(data0, dtype="int32")
w0 = relay.const(np.ndarray(shape=(30522, 768), dtype="float32"))
r1 = relay.take(w0, r0, axis=0)
r2 = relay.cast(data1, dtype="int32")
w1 = relay.const(np.ndarray(shape=(2, 768), dtype="float32"))
r3 = relay.take(w1, r2, axis=0)
r4 = relay.add(r1, r3)
r5 = relay.transpose(r4, axes=[1, 0, 2])
r6 = relay.shape_of(r5, dtype="int32")
r7 = relay.take(r6, relay.const(0, dtype="int32"))
r8 = relay.cast(r7, dtype="float32")
r9 = relay.multiply(relay.const(1, dtype="float32"), r8)
r10 = relay.add(relay.const(0, dtype="float32"), r9)
r11 = relay.arange(relay.const(0, dtype="float32"), r10,\
                   relay.const(1, dtype="float32"), dtype="float32")
r12 = relay.cast(r11, dtype="int32")
w2 = relay.const(np.ndarray(shape=(512, 768), dtype="float32"))
r13 = relay.take(w2, r12, axis=0)
r14 = relay.expand_dims(r13, axis=1)
r15 = relay.add(r5, r14)
r16 = relay.nn.dropout(r15, rate=0.1)
# r17 = relay.TupleGetItem(r16.astuple(), 0)
w3 = relay.const(np.ndarray(shape=(768,), dtype="float32"))
w4 = relay.const(np.ndarray(shape=(768,), dtype="float32"))
r18 = relay.nn.layer_norm(r16, w3, w4, epsilon=1e-12)
r19 = relay.op.reverse_reshape(r18, newshape=[-1, 0])
w5 = relay.const(np.ndarray(shape=(768, 768), dtype="float32"))
r20 = relay.reverse_reshape(w5, newshape=[12, -1, 0])
w6 = relay.const(np.ndarray(shape=(768, 768), dtype="float32"))
r21 = relay.reverse_reshape(w6, newshape=[12, -1, 0])
w7 = relay.const(np.ndarray(shape=(768, 768), dtype="float32"))
r22 = relay.reverse_reshape(w7, newshape=[12, -1, 0])
r23 = relay.Tuple([r20, r21, r22])
r24 = relay.concatenate(r23, axis=-2)
r25 = relay.reverse_reshape(r24, newshape=[-1, 0])
r26 = relay.nn.dense(r19, r25, units=2304)
w8 = relay.const(np.ndarray(shape=(768,), dtype="float32"))
w9 = relay.const(np.ndarray(shape=(768,), dtype="float32"))
w10 = relay.const(np.ndarray(shape=(768,), dtype="float32"))
r27 = relay.Tuple([w8, w9, w10])
r28 = relay.concatenate(r27, axis=0)
r29 = relay.nn.bias_add(r26, r28, axis=-1)
r30 = relay.reshape(r29, newshape=[-1, 1, 2304])
r31 = relay.reshape(r30, newshape=[0, 0, 12, 3, -1])
r32 = relay.take(r31, relay.const(0, dtype="int64"), axis=3)
r33 = relay.transpose(r32, axes=[1, 2, 0, 3])
r34 = relay.reverse_reshape(r33, newshape=[-1, 0, 0])
r35 = relay.shape_of(r34, dtype="int32")
r36 = relay.take(r35, relay.const(2, dtype="int32"))
r37 = relay.cast(r36, dtype="float32")
r38 = relay.sqrt(r37)
r39 = relay.divide(r34, r38)
r40 = relay.take(r31, relay.const(1, dtype="int64"), axis=3)
r41 = relay.transpose(r40, axes=[1, 2, 0, 3])
r42 = relay.reverse_reshape(r41, newshape=[-1, 0, 0])
r43 = relay.nn.batch_matmul(r39, r42)
# r44 = relay.nn.softmax(r43)

func = relay.Function([data0, data1], r43)
mod = tvm.ir.IRModule.from_expr(func)

params = {}
exe = relay.vm.compile(mod, target="cuda", params=params)
rt = tvm.runtime.vm.VirtualMachine(exe, tvm.gpu(0))

seq_length = 128
d0 = np.random.randint(0, 1000, size=(1, seq_length)).astype('float32')
d1 = np.ones((1, seq_length)).astype('float32')
d2 = np.asarray([seq_length]).astype('float32')

rt.set_input("main", data0=d0, data1=d1)

rt.run()
