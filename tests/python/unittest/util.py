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
from topi.util import get_const_int


def check_assert_bound(expr, var, lb, ub):
    assert isinstance(expr, tvm.expr.Call)
    assert expr.name == "tvm_assert_bound"
    assert expr.dtype == var.dtype
    assert expr.args[0] == var
    lower = get_const_int(expr.args[1]) if isinstance(expr.args[1], (tvm.expr.IntImm, tvm.expr.UIntImm)) \
                                        else expr.args[1]
    upper = get_const_int(expr.args[2]) if isinstance(expr.args[2], (tvm.expr.IntImm, tvm.expr.UIntImm)) \
                                        else expr.args[2]
    assert lower == lb
    assert upper == ub
