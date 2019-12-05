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
import os

import numpy as np

import tvm
from tvm import relay
import tvm.relay.transform as _transform

def test_eta_expand_global_var():
    mod = relay.fromtext(r"""
        v0.0.4
        def @aux(%x: Tensor[(), int32]) -> Tensor[(), int32] {
            %x
        }
        def @main() -> (fn(Tensor[(), int32]) -> Tensor[(), int32]) {
            @aux
        }
    """)
    seq = _transform.Sequential([_transform.EtaExpand(expand_global_var=True)])
    with _transform.PassContext(opt_level=3):
        mod = seq(mod)
    expected = relay.fromtext(r"""
        v0.0.4
        def @aux(%x: Tensor[(), int32]) -> Tensor[(), int32] {
            %x
        }
        def @main() -> (fn(Tensor[(), int32]) -> Tensor[(), int32]) {
            fn (%x: Tensor[(), int32]) -> Tensor[(), int32] {
                @aux(%x)
            }
        }
    """)
    relay.analysis.assert_graph_equal(mod['main'], expected['main'])


def test_eta_expand_constructor():
    mod = relay.fromtext(r"""
        v0.0.4
        type List[A] {
            Cons(A, List[A]),
            Nil,
        }
        def @main[A]() -> (fn(A, List[A]) -> List[A]) {
            Cons
        }
    """)
    seq = _transform.Sequential([_transform.EtaExpand(expand_constructor=True)])
    with _transform.PassContext(opt_level=3):
        mod = seq(mod)
    expected = relay.fromtext(r"""
        v0.0.4
        type List[A] {
            Cons(A, List[A]),
            Nil,
        }
        def @main[A]() -> (fn(A, List[A]) -> List[A]) {
            fn [A](%x: A, %xs: List[A]) -> List[A] {
                Cons(%x, %xs)
            }
        }
    """)
    relay.analysis.assert_graph_equal(mod['main'], expected['main'])


if __name__ == '__main__':
    test_eta_expand_global_var()
    test_eta_expand_constructor()
