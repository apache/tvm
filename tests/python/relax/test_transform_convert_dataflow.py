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
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R


class ExtractCompare(tvm.testing.CompareBeforeAfter):
    transform = relax.transform.ConvertToDataflow()


# functions that will not change
class TestTrivial(ExtractCompare):
    @I.ir_module
    class Before:
        # already a DF block
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            with R.dataflow():
                x = R.add(A, B)
                y = R.multiply(x, A)
                z = R.add(x, y)
                q = R.multiply(y, z)
                p = R.add(z, q)
                R.output(p)
            return p

        # too small
        @R.function
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            x = R.add(A, B)
            return x

        # too few pure ops between non-dataflow ops
        @R.function(pure=False)
        def func2(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            _ = R.print(format="Hi there!")
            y = R.add(A, B)
            _ = R.print(y, format="Sum: {}")
            x = R.multiply(y, y)
            if R.const(False):
                _ = R.print(format="True branch")
                q = R.add(x, y)
                _ = R.print(q, format="Value of q: {}")
                w = q
            else:
                _ = R.print(format="False branch")
                q = R.subtract(x, y)
                _ = R.print(q, format="Value of q: {}")
                w = q
            p = R.multiply(w, w)
            return p

    Expected = Before


class TestBasic(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v


class TestMultipleBlocks(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            _ = R.print(format="Hi mom!")
            a = R.multiply(v, v)
            b = R.add(a, a)
            c = R.subtract(b, a)
            d = R.add(c, c)
            return d

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            _ = R.print(format="Hi mom!")
            with R.dataflow():
                a = R.multiply(v, v)
                b = R.add(a, a)
                c = R.subtract(b, a)
                d = R.add(c, c)
                R.output(d)
            return d


class TestExtractInsideBranches(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            if R.const(True):
                q = R.multiply(v, v)
                a = R.add(q, q)
                b = R.multiply(a, a)
            else:
                q = R.add(v, v)
                a = R.multiply(q, q)
                b = R.add(a, a)
            c = R.multiply(b, b)
            d = R.add(c, c)
            e = R.multiply(d, d)
            return e

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)

            if R.const(True):
                with R.dataflow():
                    q = R.multiply(v, v)
                    a = R.add(q, q)
                    b = R.multiply(a, a)
                    R.output(b)
                # weird but the parser requires this construct
                c = b
            else:
                with R.dataflow():
                    q = R.add(v, v)
                    a = R.multiply(q, q)
                    b = R.add(a, a)
                    R.output(b)
                c = b
            with R.dataflow():
                d = R.multiply(c, c)
                e = R.add(d, d)
                f = R.multiply(e, e)
                R.output(f)
            return f


class TestTreatNonCallAsPure(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(t: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
            x = t[0]
            y = t[1]
            z = R.add(x, y)
            w = R.multiply(z, z)
            return w

    @I.ir_module
    class Expected:
        @R.function
        def main(t: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
            with R.dataflow():
                x = t[0]
                y = t[1]
                z = R.add(x, y)
                w = R.multiply(z, z)
                R.output(w)
            return w


class TestInnerFunction(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            @R.function(pure=False)
            def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                z = R.add(x, y)
                w = R.multiply(x, z)
                v = R.add(y, w)
                _ = R.print(format="oops")
                a = R.multiply(v, v)
                b = R.add(a, a)
                c = R.multiply(a, b)
                return c

            z = R.add(x, y)
            w = R.multiply(z, z)
            v = R.divide(w, z)
            q = inner_func(w, v)
            a = R.multiply(q, q)
            b = R.add(a, a)
            c = R.multiply(b, a)
            return c

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():

                @R.function(pure=False)
                def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                    with R.dataflow():
                        z = R.add(x, y)
                        w = R.multiply(x, z)
                        v = R.add(y, w)
                        R.output(v)
                    _ = R.print(format="oops")
                    with R.dataflow():
                        a = R.multiply(v, v)
                        b = R.add(a, a)
                        c = R.multiply(a, b)
                        R.output(c)
                    return c

                z = R.add(x, y)
                w = R.multiply(z, z)
                v = R.divide(w, z)
                R.output(inner_func, v, w)
            q = inner_func(w, v)
            with R.dataflow():
                a = R.multiply(q, q)
                b = R.add(a, a)
                c = R.multiply(b, a)
                R.output(c)
            return c


class TestMergeWithPrecedingDataflowBlock(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                R.output(w)

            # The single binding of `v = R.add` would normally not be
            # enough to make a dataflow block, as `1 < min_size == 2`.
            v = R.add(w, x)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                # However, it occurs just after an existing dataflow
                # block, and can be merged into it.
                v = R.add(w, x)
                R.output(v)
            return v


class TestMergeWithNextDataflowBlock(ExtractCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            # The single binding of `z = R.add` would normally not be
            # enough to make a dataflow block, as `1 < min_size == 2`.
            z = R.add(x, y)

            # However, it occurs just before an existing dataflow
            # block, and can be merged into it.

            with R.dataflow():
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v


if __name__ == "__main__":
    tvm.testing.main()
