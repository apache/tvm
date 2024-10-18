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

import argparse
import pytest
from tvm.driver.tvmc.pass_list import parse_pass_list_str


def test_parse_pass_list_str():
    assert [""] == parse_pass_list_str("")
    assert ["FoldScaleAxis", "FuseOps"] == parse_pass_list_str("FoldScaleAxis,FuseOps")
    assert ["tir.UnrollLoop", "qnn.Legalize"] == parse_pass_list_str("tir.UnrollLoop,qnn.Legalize")

    with pytest.raises(argparse.ArgumentTypeError) as ate:
        parse_pass_list_str("MyYobaPass,qnn.MySuperYobaPass,FuseOps")

    assert "MyYobaPass" in str(ate.value)
    assert "qnn.MySuperYobaPass" in str(ate.value)
    assert "FuseOps" in str(ate.value)


if __name__ == "__main__":
    tvm.testing.main()
