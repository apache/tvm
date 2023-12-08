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

from tvm import autotvm
import tvm


@autotvm.template("testing/dispatch_fallback")
def simple_template(a, b):
    cfg = autotvm.get_config()
    assert cfg.is_fallback


def test_fallback():
    simple_template(2, 3)


def test_tophub_kinds_match():
    def verify_arm_cpu(target):
        best_by_targetkey = autotvm.tophub.context(target).best_by_targetkey
        assert len(best_by_targetkey)
        found_arm_cpu = False
        for a, _ in best_by_targetkey:
            if "arm_cpu" in a:
                found_arm_cpu = True
                break
        assert found_arm_cpu

    verify_arm_cpu("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod")
    verify_arm_cpu("llvm -model=snapdragon835 -mtriple=arm64-linux-android -mattr=+neon")


if __name__ == "__main__":
    test_fallback()
