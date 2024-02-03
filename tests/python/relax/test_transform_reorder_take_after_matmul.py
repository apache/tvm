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

import inspect

import pytest

import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T


class Base:
    def test_compare(self):
        transform = relax.transform.ReorderTakeAfterMatmul()

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm.ir.assert_structural_equal(self.Expected, after)


class TestSimple(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, "weight_table_size"], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            weight_table_size = T.int64()
            with R.dataflow():
                weight: R.Tensor([16, 32], "float32") = R.take(weight_table, routing_table, axis=1)
                out: R.Tensor([1, 32], "float32") = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 16], "float32"),
            weight_table: R.Tensor([16, "weight_table_size"], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor([1, 32], "float32"):
            weight_table_size = T.int64()
            with R.dataflow():
                out_table: R.Tensor([1, weight_table_size], "float32") = R.matmul(x, weight_table)
                out: R.Tensor([1, 32], "float32") = R.take(out_table, routing_table, axis=1)
                R.output(out)
            return out


class TestBatchedActivations(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight_table: R.Tensor([16, "weight_table_size"], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            batch_size = T.int64()
            weight_table_size = T.int64()
            with R.dataflow():
                weight: R.Tensor([16, 32], "float32") = R.take(weight_table, routing_table, axis=1)
                out: R.Tensor([batch_size, 1, 32], "float32") = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight_table: R.Tensor([16, "weight_table_size"], "float32"),
            routing_table: R.Tensor([32], "int64"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            batch_size = T.int64()
            weight_table_size = T.int64()
            with R.dataflow():
                out_table: R.Tensor([batch_size, 1, weight_table_size], "float32") = R.matmul(
                    x, weight_table
                )
                out: R.Tensor([batch_size, 1, 32], "float32") = R.take(
                    out_table, routing_table, axis=2
                )
                R.output(out)
            return out


class TestStaticBatchedActivationsAndWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            batch_size = T.int64()
            routing_table_size = T.int64()
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([128, 1, 16], "float32"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
            routing_table: R.Tensor([128], "int64"),
        ) -> R.Tensor([128, 1, 32], "float32"):
            batch_size = T.int64()
            routing_table_size = T.int64()
            with R.dataflow():
                reordered_weight = R.permute_dims(weight_table, [1, 0, 2])
                fused_weight = R.reshape(reordered_weight, [16, routing_table_size * 32])
                fused_output = R.matmul(x, fused_weight)
                reordered_output = R.reshape(fused_output, [128, 1, routing_table_size, 32])
                tabular_output = R.take(reordered_output, routing_table, axis=2)
                out = R.einsum([tabular_output], "ijik->ijk")
                R.output(out)
            return out


class TestDynamicBatchedActivationsAndWeights(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
            routing_table: R.Tensor(["batch_size"], "int64"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            batch_size = T.int64()
            routing_table_size = T.int64()
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                out = R.matmul(x, weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
            routing_table: R.Tensor(["batch_size"], "int64"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            batch_size = T.int64()
            routing_table_size = T.int64()
            with R.dataflow():
                reordered_weight = R.permute_dims(weight_table, [1, 0, 2])
                fused_weight = R.reshape(reordered_weight, [16, routing_table_size * 32])
                fused_output = R.matmul(x, fused_weight)
                reordered_output = R.reshape(fused_output, [batch_size, 1, routing_table_size, 32])
                tabular_output = R.take(reordered_output, routing_table, axis=2)
                out = R.einsum([tabular_output], "ijik->ijk")
                R.output(out)
            return out


if __name__ == "__main__":
    tvm.testing.main()
