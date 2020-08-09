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

from tvm import register_func, get_global_func, IRModule
from tvm import relay
from tvm.parser import SpanCheck
from tvm.parser import AnnotateSpans
from tvm.runtime import Object
from tvm.ir.diagnostic import get_default_renderer, set_default_renderer

# std_out = get_default_renderer()()


# def testing_renderer(diag_ctx):
#     std_out.render(diag_ctx)
#     return


# set_default_renderer(testing_renderer)


# def test_span_check():
#     data = relay.var("data", shape=(10, 1, 1, 1))
#     weight = relay.var("weight", shape=(7, 5, 6, 7))
#     conv = relay.nn.conv2d(data, weight)
#     mod = IRModule.from_expr(conv)
#     print(mod["main"])
#     mod = AnnotateSpans()(mod)
#     # mod = SpanCheck()(mod)


# def test_parser_span():
#     pass


# test_span_check()
