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
"""Test dispatcher.
The dispatcher can choose which template to use according
to the parameters of workload"""

from collections import namedtuple
from tvm import autotvm
from tvm.autotvm.task import dispatcher, DispatchContext

SimpleConfig = namedtuple('SimpleConfig', ('template_key', 'is_fallback'))

def test_dispatch():
    @dispatcher
    def my_dispatcher(a, b):
        return (a, b)

    @my_dispatcher.register("im2col")
    def _im2col(cfg, a, b):
        return a

    @my_dispatcher.register("spatial_pack")
    def _spatial_pack(cfg, a, b):
        return b

    class SimpleDispatcher(DispatchContext):
        def query(self, target, workload):
            a, b = workload
            tkey = "spatial_pack" if a + b > 2 else "im2col"
            cfg = SimpleConfig(tkey, False)
            return cfg

    with SimpleDispatcher():
        # this will call im2col
        assert my_dispatcher(1, 0) == 1

        # this will call spatial pack
        assert my_dispatcher(1, 100) == 100

def test_fallback():

    @autotvm.template
    def simple_template(a, b):
        cfg = autotvm.get_config()
        assert cfg.is_fallback

    simple_template(2, 3)


if __name__ == "__main__":
    test_dispatch()
    test_fallback()
