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
import tvm.testing
from tvm.driver.tvmc import TVMCException
from tvm.driver.tvmc.target import target_from_cli, tokenize_target, parse_target


def test_target_from_cli__error_duplicate():
    with pytest.raises(TVMCException):
        _ = target_from_cli("llvm, llvm")


def test_target_invalid_more_than_two_tvm_targets():
    with pytest.raises(TVMCException):
        _ = target_from_cli("cuda, opencl, llvm")


def test_target_from_cli__error_target_not_found():
    with pytest.raises(TVMCException):
        _ = target_from_cli("invalidtarget")


def test_target_two_tvm_targets():
    tvm_target, extra_targets = target_from_cli(
        "opencl -device=mali, llvm -mtriple=aarch64-linux-gnu"
    )

    assert "opencl" in str(tvm_target)
    assert "llvm" in str(tvm_target.host)

    # No extra targets
    assert 0 == len(extra_targets)


def test_tokenize_target_with_opts():
    tokens = tokenize_target("foo -opt1=value1 --flag, bar -opt2=value2")
    expected_tokens = ["foo", "-opt1=value1", "--flag", ",", "bar", "-opt2=value2"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_plus_sign():
    tokens = tokenize_target("foo -opt1=+value1 --flag, bar -opt2=test,+v")
    expected_tokens = ["foo", "-opt1=+value1", "--flag", ",", "bar", "-opt2=test,+v"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas():
    tokens = tokenize_target("foo -opt1=v,a,l,u,e,1 --flag")
    expected_tokens = ["foo", "-opt1=v,a,l,u,e,1", "--flag"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas_and_single_quotes():
    tokens = tokenize_target("foo -opt1='v, a, l, u, e', bar")
    expected_tokens = ["foo", "-opt1='v, a, l, u, e'", ",", "bar"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas_and_double_quotes():
    tokens = tokenize_target('foo -opt1="v, a, l, u, e", bar')
    expected_tokens = ["foo", '-opt1="v, a, l, u, e"', ",", "bar"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_dashes():
    tokens = tokenize_target("foo-bar1 -opt-1=t-e-s-t, baz")
    expected_tokens = ["foo-bar1", "-opt-1=t-e-s-t", ",", "baz"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_single_target_with_opts():
    targets = parse_target("llvm -device=arm_cpu -mattr=+fp")

    assert len(targets) == 1
    assert "device" in targets[0]["opts"]
    assert "mattr" in targets[0]["opts"]


def test_parse_multiple_target():
    targets = parse_target("compute-library, llvm -device=arm_cpu")

    assert len(targets) == 2
    assert "compute-library" == targets[0]["name"]
    assert "llvm" == targets[1]["name"]


def test_parse_quotes_and_separators_on_options():
    targets_no_quote = parse_target("foo -option1=+v1.0x,+value,+bar")
    targets_single_quote = parse_target("foo -option1='+v1.0x,+value'")
    targets_double_quote = parse_target('foo -option1="+v1.0x,+value"')

    assert len(targets_no_quote) == 1
    assert "+v1.0x,+value,+bar" == targets_no_quote[0]["opts"]["option1"]

    assert len(targets_single_quote) == 1
    assert "+v1.0x,+value" == targets_single_quote[0]["opts"]["option1"]

    assert len(targets_double_quote) == 1
    assert "+v1.0x,+value" == targets_double_quote[0]["opts"]["option1"]


if __name__ == "__main__":
    tvm.testing.main()
