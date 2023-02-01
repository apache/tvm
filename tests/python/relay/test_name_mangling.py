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
# under the License

import tvm
import tvm.testing
import tvm.relay as relay
import tvm.relay.backend.utils as utils
import pytest


def test_mangle_mod_name():
    assert utils.mangle_module_name("default") == "tvmgen_default"
    assert utils.mangle_module_name("ccompiler") == "tvmgen_ccompiler"
    assert utils.mangle_module_name("1234"), "tvmgen_1234"
    assert utils.mangle_module_name(""), "tvmgen"
    assert utils.mangle_module_name(None), "tvmgen"

    with pytest.raises(ValueError):
        utils.mangle_module_name("\u018e")
        utils.mangle_module_name("\xf1")


if __name__ == "__main__":
    tvm.testing.main()
