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
# pylint: disable=invalid-name, unused-argument, no-else-return, E1102
"""Torch codegen operators"""

from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end


def torchop(script_fn, *params):
    """Insert an Operation executed in the PyTorch JIT

    The operation includes backend annotation

    Currently, only tensors are supported. The shape inferrence
    assumes that input shapes (and not values) determine output shapes."""
    return compiler_end(
        relay.op._make.torchop(
            [compiler_begin(p, "torch") for p in params], script_fn.save_to_buffer()
        ),
        "torch",
    )
