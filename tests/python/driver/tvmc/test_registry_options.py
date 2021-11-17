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

from tvm.driver.tvmc import TVMCException
from tvm.driver.tvmc.registry import generate_registry_args, reconstruct_registry_entity
from tvm.relay.backend import Executor


def test_registry_to_argparse():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor)
    parsed, _ = parser.parse_known_args(["--executor=aot", "--executor-aot-interface-api=c"])

    assert parsed.executor == "aot"
    assert parsed.executor_aot_interface_api == "c"


def test_registry_to_argparse_default():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor, "aot")
    parsed, _ = parser.parse_known_args([])

    assert parsed.executor == "aot"


def test_mapping_registered_args():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor)
    parsed, _ = parser.parse_known_args(["--executor=aot", "--executor-aot-interface-api=c"])
    entity = reconstruct_registry_entity(parsed, Executor)

    assert isinstance(entity, Executor)
    assert "interface-api" in entity
    assert entity["interface-api"] == "c"


def test_mapping_registered_args_no_match_for_name():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor)
    parsed, _ = parser.parse_known_args(["--executor=woof"])

    with pytest.raises(TVMCException, match='Executor "woof" is not defined'):
        reconstruct_registry_entity(parsed, Executor)


def test_mapping_registered_args_no_arg():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor)
    parsed, _ = parser.parse_known_args([])

    assert reconstruct_registry_entity(parsed, Executor) == None


def test_mapping_registered_args_mismatch_for_arg():
    parser = argparse.ArgumentParser()
    generate_registry_args(parser, Executor)
    parsed, _ = parser.parse_known_args(["--executor=aot", "--executor-graph-link-params=1"])

    with pytest.raises(
        TVMCException,
        match="Passed --executor-graph-link-params but did not specify graph executor",
    ):
        reconstruct_registry_entity(parsed, Executor)
