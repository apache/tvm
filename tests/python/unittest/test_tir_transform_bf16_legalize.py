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
import topi
from tvm import te
from tvm.tir import const


def lower_stmt(sche, params, passfunc):
    func = tvm.driver.build_module.form_irmodule(sche, params, "main", None)["main"]
    func = passfunc()(
        tvm.IRModule.from_expr(func))["main"]
    stmt = func.body
    return stmt

def to32(v):
    return topi.cast(v, 'float')
def to16(v):
    return topi.cast(v, 'bf16')

def test_promote():
    def runpass(op, passfunc):
        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), lambda i: op(a[i], b[i]))
        s = te.create_schedule(c.op)
        return lower_stmt(s, [a, b, c], passfunc)
    
    def get_promoted(op):
        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), lambda i:
                topi.cast(op(topi.cast(a[i],'float'),
                    topi.cast(b[i],'float')), 'bf16')
                )
        s = te.create_schedule(c.op)
        func = tvm.driver.build_module.form_irmodule(s, [a,b,c], "main", None)["main"]
        return func.body

    def test_promoted(op):
        stmt = runpass(op, tvm.tir.transform.BF16Promote)
        tvm.ir.assert_structural_equal(stmt, get_promoted(op))
    test_promoted(topi.add)
    test_promoted(topi.subtract)
    test_promoted(topi.multiply)
    test_promoted(topi.divide)

def test_eliminate():
    def get_eliminated():
        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), lambda i: to16(
            topi.add(
                to32(
                    to16(
                        topi.add(
                            to32(a[i]),
                            to32(b[i]),
                        )
                    )
                ),
                to32(
                    to16(
                        topi.add(
                            to32(a[i]),
                            to32(b[i]),
                        )
                    )
                )
            )
        ))
        s = te.create_schedule(c.op)
        stmt = lower_stmt(s, [a, b, c], tvm.tir.transform.BF16CastElimination)
        return stmt

    def get_target():
        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), lambda i: to16(
            topi.add(topi.add(
                        to32(a[i]),
                        to32(b[i]),
                    ),
                    topi.add(
                        to32(a[i]),
                        to32(b[i]),
                    )
                )
        ))
        s = te.create_schedule(c.op)
        func = tvm.driver.build_module.form_irmodule(s, [a,b,c], "main", None)["main"]
        return func.body

    tvm.ir.assert_structural_equal(get_eliminated(), get_target())

def test_legalize():
    def check(fcompute_before, fcompute_after):
        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), fcompute_before(a,b))
        s = te.create_schedule(c.op)
        stmt = lower_stmt(s, [a, b, c], tvm.tir.transform.BF16Legalize)

        a = te.placeholder((100,), dtype='bf16')
        b = te.placeholder((100,), dtype='bf16')
        c = te.compute((100,), fcompute_after(a,b))
        s = te.create_schedule(c.op)
        func = tvm.driver.build_module.form_irmodule(s, [a,b,c], "main", None)["main"]
        tvm.ir.assert_structural_equal(stmt, func.body)

    def orig1(a,b):
        return lambda i: a[i]+b[i]+a[99-i]+b[99-i]
    def after1(a,b):
        return lambda i: to16(to32(a[i])+to32(b[i])+to32(a[99-i])+to32(b[99-i]))
    def orig1(a,b):
        return lambda i: a[i]*b[i]+a[99-i]*b[99-i]+a[i]
    def after1(a,b):
        return lambda i: to16(to32(a[i])*to32(b[i])+to32(a[99-i])*to32(b[99-i])+to32(a[i]))

    check(orig1, after1)

if __name__ == "__main__":
    test_promote()
    test_eliminate()
    test_legalize()