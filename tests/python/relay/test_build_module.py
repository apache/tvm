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

from tvm.target.target import Target
from tvm.relay.backend import Runtime, Executor
from tvm.relay.build_module import _reconstruct_from_deprecated_options


@pytest.mark.parametrize(
    "target,executor,runtime",
    [
        [Target("c"), None, None],
        [Target("c -runtime=c"), None, Runtime("crt")],
        [Target("c -system-lib"), None, Runtime("cpp", {"system-lib": True})],
        [Target("c -runtime=c -system-lib"), None, Runtime("crt", {"system-lib": True})],
        [Target("c -executor=aot"), Executor("aot"), None],
        [
            Target("c -executor=aot -interface-api=c"),
            Executor("aot", {"interface-api": "c"}),
            None,
        ],
        [
            Target("c -executor=aot -unpacked-api=1"),
            Executor("aot", {"unpacked-api": True}),
            None,
        ],
        [Target("c -executor=aot -link-params=1"), Executor("aot"), None],
        [Target("c -link-params=1"), Executor("graph", {"link-params": True}), None],
        [
            Target(
                "c -executor=aot -link-params=1 -interface-api=c"
                "  -unpacked-api=1 -runtime=c -system-lib"
            ),
            Executor("aot", {"unpacked-api": True, "interface-api": "c"}),
            Runtime("crt", {"system-lib": True}),
        ],
    ],
)
def test_deprecated_target_parameters(target, executor, runtime):
    actual_executor, actual_runtime = _reconstruct_from_deprecated_options(target)
    assert executor == actual_executor
    assert runtime == actual_runtime


if __name__ == "__main__":
    pytest.main()
