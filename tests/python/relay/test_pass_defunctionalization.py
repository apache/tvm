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

import tvm
from tvm import relay
from tvm.relay.transform import Defunctionalization, InferType, LambdaLift

def test_simple():
    code = """
#[version = "0.0.5"]
def @apply[A, B](%f: fn(A) -> B, %xs: A) -> B {
  %f(%xs)
}
def @main(%l: float32) -> float32 {
  %0 = fn[A](%x: A) -> A {
    %x
  };
  @apply(%0, %l)
}
"""
    mod = tvm.parser.fromtext(code)
    mod = LambdaLift()(mod)
    mod = InferType()(mod)
    expr = Defunctionalization(mod['main'], mod)

def test_global_recursion():
  code = """
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}
def @id[A](%x: A) -> A {
  %x
}
def @map[A, B](%f: fn(A) -> B, %xs: List[A]) -> List[B] {
  match (%xs) {
    Cons(%x, %rest) => Cons(%f(%x), @map(%f, %rest)),
    Nil => Nil,
  }
}
def @main(%l: List[float32]) -> List[float32] {
  @map(@id, %l)
}
"""
  mod = tvm.parser.fromtext(code)
  # mod = LambdaLift()(mod)
  mod = InferType()(mod)
  # expr = Defunctionalization(mod['main'], mod)

def test_sum():
  code = """
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}
def @main(%f: fn(int32) -> int32, %xs: List[int32]) -> int32 {
  match (%xs) {
    Cons(%x, %rest) => %0 = fn(%n) {
      %x + %f(%n)
    };
    @main(%0, %rest),
    Nil => %f(0),
  }
}
"""
  mod = tvm.parser.fromtext(code)
  mod = LambdaLift()(mod)
  mod = InferType()(mod)
  print(mod)

def test():
  code = """
#[version = "0.0.5"]
def @id[A](%x: A) -> A {
  %x
}
def @main(%f: float32) -> float32 {
  @id(@id)(%f)
}
"""
  mod = tvm.parser.fromtext(code)
  mod = InferType()(mod)
  print(mod['main'].body.type_args)

if __name__ == "__main__":
  # pytest.main([__file__])
  #   test_simple()
  #   test_global_recursion()
    test_global_recursion()