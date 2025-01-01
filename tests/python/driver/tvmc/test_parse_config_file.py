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
import os
import shlex

import tvm
from tvm.driver.tvmc.main import _main
from tvm.driver.tvmc.config_options import convert_config_json_to_cli, get_configs_json_dir


def test_parse_json_config_file_one_target():
    tokens = convert_config_json_to_cli(
        {"targets": [{"kind": "llvm"}], "output": "resnet50-v2-7-autotuner_records.json"}
    )
    expected_tokens = [{"target": "llvm"}, {"output": "resnet50-v2-7-autotuner_records.json"}]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_json_config_file_multipile_targets():
    tokens = convert_config_json_to_cli(
        {
            "targets": [{"kind": "llvm"}, {"kind": "c", "mcpu": "cortex-m55"}],
            "tuning-records": "resnet50-v2-7-autotuner_records.json",
            "pass-config": {"tir.disable_vectorizer": "1"},
        }
    )
    expected_tokens = [
        {"target_c_mcpu": "cortex-m55"},
        {"target": "llvm, c"},
        {"tuning_records": "resnet50-v2-7-autotuner_records.json"},
        {"pass_config": ["tir.disable_vectorizer=1"]},
    ]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_json_config_file_executor():
    tokens = convert_config_json_to_cli(
        {
            "executor": {"kind": "aot", "interface-api": "c"},
            "inputs": "imagenet_cat.npz",
            "max-local-memory-per-block": "4",
            "repeat": "100",
        }
    )
    expected_tokens = [
        {"executor": "aot"},
        {"executor_aot_interface_api": "c"},
        {"inputs": "imagenet_cat.npz"},
        {"max_local_memory_per_block": "4"},
        {"repeat": "100"},
    ]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_json_config_file_target_and_executor():
    tokens = convert_config_json_to_cli(
        {
            "targets": [
                {"kind": "ethos-u -accelerator_config=ethos-u55-256"},
                {"kind": "c", "mcpu": "cortex-m55"},
                {"kind": "cmsis-nn"},
            ],
            "executor": {"kind": "aot", "interface-api": "c", "unpacked-api": "1"},
            "inputs": "imagenet_cat.npz",
            "max-local-memory-per-block": "4",
            "repeat": "100",
        }
    )
    expected_tokens = [
        {"target_c_mcpu": "cortex-m55"},
        {"target": "ethos-u -accelerator_config=ethos-u55-256, c, cmsis-nn"},
        {"executor": "aot"},
        {"executor_aot_interface_api": "c"},
        {"executor_aot_unpacked_api": "1"},
        {"inputs": "imagenet_cat.npz"},
        {"max_local_memory_per_block": "4"},
        {"repeat": "100"},
    ]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_json_config_file_runtime():
    tokens = convert_config_json_to_cli(
        {
            "targets": [
                {"kind": "cmsis-nn", "from_device": "1"},
                {"kind": "c", "mcpu": "cortex-m55"},
            ],
            "runtime": {"kind": "crt"},
            "inputs": "imagenet_cat.npz",
            "output": "predictions.npz",
            "pass-config": {"tir.disable_vectorize": "1", "relay.backend.use_auto_scheduler": "0"},
        }
    )
    expected_tokens = [
        {"target_cmsis-nn_from_device": "1"},
        {"target_c_mcpu": "cortex-m55"},
        {"target": "cmsis-nn, c"},
        {"runtime": "crt"},
        {"inputs": "imagenet_cat.npz"},
        {"output": "predictions.npz"},
        {"pass_config": ["tir.disable_vectorize=1", "relay.backend.use_auto_scheduler=0"]},
    ]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tvmc_get_configs_json_dir(tmpdir_factory, monkeypatch):
    # Reset global state
    monkeypatch.setattr(tvm.driver.tvmc.config_options, "CONFIGS_JSON_DIR", None)

    # Get default directory for reference
    default_dir = get_configs_json_dir()

    # Set custom dir which does not exist -> ignore
    monkeypatch.setattr(tvm.driver.tvmc.config_options, "CONFIGS_JSON_DIR", None)
    monkeypatch.setenv("TVM_CONFIGS_JSON_DIR", "not_a_directory")
    result = get_configs_json_dir()
    assert_msg = "Non-existent directory passed via TVM_CONFIGS_JSON_DIR should be ignored."
    assert result == default_dir, assert_msg

    # Set custom dir which does exist
    monkeypatch.setattr(tvm.driver.tvmc.config_options, "CONFIGS_JSON_DIR", None)
    configs_dir = tmpdir_factory.mktemp("configs")
    monkeypatch.setenv("TVM_CONFIGS_JSON_DIR", str(configs_dir))
    result = get_configs_json_dir()
    assert_msg = (
        "Custom value passed via TVM_CONFIGS_JSON_DIR should be used instead of default one."
    )
    assert result != default_dir and result is not None, assert_msg
