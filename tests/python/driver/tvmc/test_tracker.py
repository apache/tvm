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

from tvm.driver.tvmc.tracker import tracker_host_port_from_cli


def test_tracker_host_port_from_cli__hostname_port():
    input_str = "1.2.3.4:9090"
    expected_host = "1.2.3.4"
    expected_port = 9090

    actual_host, actual_port = tracker_host_port_from_cli(input_str)

    assert expected_host == actual_host
    assert expected_port == actual_port


def test_tracker_host_port_from_cli__hostname_port__empty():
    input_str = ""

    actual_host, actual_port = tracker_host_port_from_cli(input_str)

    assert actual_host is None
    assert actual_port is None


def test_tracker_host_port_from_cli__only_hostname__default_port_is_9090():
    input_str = "1.2.3.4"
    expected_host = "1.2.3.4"
    expected_port = 9090

    actual_host, actual_port = tracker_host_port_from_cli(input_str)

    assert expected_host == actual_host
    assert expected_port == actual_port
