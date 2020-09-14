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
from tvm import te
import os


def test_plugin_module():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    mod = tvm.runtime.load_module(os.path.join(curr_path, "lib", "plugin_module.so"))
    # NOTE: we need to make sure all managed resources returned
    # from mod get destructed before mod get unloaded.
    #
    # Failure mode we want to prevent from:
    # We retain an object X whose destructor is within mod.
    # The program will segfault if X get destructed after mod,
    # because the destructor function has already been unloaded.
    #
    # The easiest way to achieve this is to wrap the
    # logics related to mod inside a function.
    def run_module(mod):
        # normal functions
        assert mod["AddOne"](10) == 11
        assert mod["SubOne"](10) == 9
        # advanced usecase: return a module
        mymod = mod["CreateMyModule"](10)
        fadd = mymod["add"]
        assert fadd(10) == 20
        assert mymod["mul"](10) == 100

    run_module(mod)


if __name__ == "__main__":
    test_plugin_module()
