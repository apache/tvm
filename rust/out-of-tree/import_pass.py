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
import tvm.relay
from tvm.ir.transform import PassContext

x = tvm.relay.var("x", shape=(10,))
test_func = tvm.relay.Function([x], x)
test_mod = tvm.IRModule.from_expr(test_func)

pass_dylib = "/Users/jroesch/Git/tvm/rust/target/debug/libmy_pass.dylib"

def load_rust_extension(ext_dylib):
    load_so = tvm.get_global_func("runtime.module.loadfile_so")
    mod = load_so(ext_dylib)
    mod.get_function("initialize")()


def load_pass(pass_name, dylib):
    load_rust_extension(dylib)
    return tvm.get_global_func(pass_name)

MyPass = load_pass("out_of_tree.Pass", pass_dylib)
ctx = PassContext()
import pdb; pdb.set_trace()
f = MyPass(test_func, test_mod, ctx)
mod = MyPass()(test_mod)

print(mod)
