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
import tvm.testing
from tvm.ir.supply import NameSupply


def test_fresh_name_empty_string():
    """Empty name should produce a valid variable name, not an empty string."""
    ns = NameSupply("")
    name = ns.fresh_name("", add_prefix=False)
    assert name == "v"
    name2 = ns.fresh_name("", add_prefix=False)
    assert name2 == "v_1"


def test_fresh_name_empty_string_with_prefix():
    """Empty name with prefix should produce a valid variable name."""
    ns = NameSupply("prefix")
    name = ns.fresh_name("", add_prefix=True)
    assert name == "prefix_v"
    name2 = ns.fresh_name("", add_prefix=True)
    assert name2 == "prefix_v_1"


if __name__ == "__main__":
    tvm.testing.main()
