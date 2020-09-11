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
import vta


def test_env():
    env = vta.get_env()
    mock = env.mock
    assert mock.alu == "skip_alu"


def test_env_scope():
    env = vta.get_env()
    cfg = env.cfg_dict
    cfg["TARGET"] = "xyz"
    with vta.Environment(cfg):
        assert vta.get_env().TARGET == "xyz"
    assert vta.get_env().TARGET == env.TARGET


if __name__ == "__main__":
    test_env()
    test_env_scope()
