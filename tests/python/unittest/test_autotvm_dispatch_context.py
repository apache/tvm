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


def test_fallback():
    @autotvm.template("testing/dispatch_fallback")
    def simple_template(a, b):
        cfg = autotvm.get_config()
        assert cfg.is_fallback

    simple_template(2, 3)


if __name__ == "__main__":
    test_fallback()
