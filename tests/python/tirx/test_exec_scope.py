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

from tvm.tirx.exec_scope import ExecScope


def test_exec_scope_create():
    def is_trivial_scope(scope, name):
        return isinstance(scope, ExecScope) and scope.name == name

    thread = ExecScope("thread")
    warp = ExecScope("warp")
    wg = ExecScope("warpgroup")
    cta = ExecScope("cta")
    cluster = ExecScope("cluster")
    kernel = ExecScope("kernel")
    world = ExecScope("world")

    assert is_trivial_scope(world, "world")
    assert is_trivial_scope(kernel, "kernel")
    assert is_trivial_scope(thread, "thread")
    assert is_trivial_scope(warp, "warp")
    assert is_trivial_scope(wg, "warpgroup")
    assert is_trivial_scope(cta, "cta")
    assert is_trivial_scope(cluster, "cluster")

    with pytest.raises(Exception, match="Unknown scope kind name"):
        ExecScope("aaa")


if __name__ == "__main__":
    test_exec_scope_create()
