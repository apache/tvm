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
from typing import Optional

import pytest

import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R


class Base:
    def test_compare(self):
        transform = relax.transform.UpdateParamStructInfo(self.update_sinfo)

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm.ir.assert_structural_equal(self.Expected, after)

    def update_sinfo(self, var: relax.Var) -> Optional[relax.StructInfo]:
        """The struct info update function provided to the transform"""
        raise NotImplementedError("Should be implemented in derived class")


class TestSimple(Base):
    def update_sinfo(self, var: relax.Var) -> Optional[relax.StructInfo]:
        if var.name_hint == "weight":
            return relax.TensorStructInfo([64, 16], "float32")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out: R.Tensor([32], "float32") = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([64, 16], "float32"),
        ) -> R.Tensor([64], "float32"):
            out: R.Tensor([64], "float32") = R.matmul(weight, x)
            return out


if __name__ == "__main__":
    tvm.testing.main()
