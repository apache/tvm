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
import pytest
from tvm.relay.transform import CollagePartition, InferType, CapturePostDfsIndexInSpans
from tvm.target import make_compilation_config
from tvm.relay.collage import MockCostEstimator
from unittest.mock import patch
from tvm.relay.dataflow_pattern import is_op, wildcard


# We'll reuse the target kind "example_target_hook" (registered in
# src/relay/backend/contrib/example_target_hooks/target.cc) as our
# example external codegen target.


def test_pattern_table():
    def relu_pattern():
        return is_op("nn.relu")(wildcard())

    def add_pattern():
        return is_op("add")(wildcard(), wildcard())

    def concatenate_pattern():
        return is_op("concatenate")(wildcard())

    def predicate(expr):
        return True

    return [
        ("relu", relu_pattern(), predicate),
        ("add", add_pattern(), predicate),
        ("concatenate", concatenate_pattern(), predicate),
    ]


def _mock_get_pattern_table(target):
    if target == "example_target_hook":
        return test_pattern_table()


def run_collage(
    input_mod, targets, cost_estimator, expected_mod, tvm_max_depth=8, byoc_max_depth=8
):
    ctxt = {
        "relay.collage.tvm_max_depth": tvm_max_depth,
        "relay.collage.byoc_max_depth": byoc_max_depth,
    }
    expected_mod = InferType()(expected_mod)
    pass_ctxt = tvm.transform.PassContext(config=ctxt)
    with pass_ctxt:
        config = make_compilation_config(pass_ctxt, targets)
        actual_mod = InferType()(input_mod)
        # Capture indexes only to help debug failing tests
        actual_mod = CapturePostDfsIndexInSpans()(actual_mod)
        actual_mod = CollagePartition(config, cost_estimator)(actual_mod)

        if not tvm.ir.structural_equal(actual_mod, expected_mod, map_free_vars=True):
            # Print everything in full so we can see what's going on when things fail.
            print("Input module:")
            print(input_mod)
            print("Actual module:")
            print(actual_mod)
            print("Expected module:")
            print(expected_mod)
            # Assert again so as to see the actual disagreeing sub-expressions.
            tvm.ir.assert_structural_equal(actual_mod, expected_mod, map_free_vars=True)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_partition_single_op_llvm(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        nn.relu(%x)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        nn.relu(%x)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 1,
            "example_target_hook": 2,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_partition_single_op_byoc(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        nn.relu(%x)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_01)
        };
        %0(%FunctionVar_0)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        @collage_example_target_hook_nn_relu(%x)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod)


@pytest.mark.parametrize("byoc_max_depth", [1, 3])
@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_partition_diamond_valid_topology(mock_get_pattern_table, byoc_max_depth):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = abs(%0);
        %2 = nn.relu(%1);
        add(%1, %2)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_3_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_01)
        };
        %0(%FunctionVar_0)
      }

      def @collage_example_target_hook_nn_relu_add(%FunctionVar_02: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu_add") -> Tensor[(10, 10), float32] {
        %1 = fn (%FunctionVar_04: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_04)
        };
        %2 = %1(%FunctionVar_02);
        %3 = fn (%FunctionVar_03: Tensor[(10, 10), float32], %FunctionVar_1: Tensor[(10, 10), float32], Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_03, %FunctionVar_1)
        };
        %3(%FunctionVar_02, %2)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        %4 = @collage_example_target_hook_nn_relu(%x);
        %5 = abs(%4);
        @collage_example_target_hook_nn_relu_add(%5)
      }
    """
    expected_1_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_02)
        };
        %1 = %0(%FunctionVar_0);
        %2 = fn (%FunctionVar_01: Tensor[(10, 10), float32], %FunctionVar_1: Tensor[(10, 10), float32], Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_01, %FunctionVar_1)
        };
        %2(%FunctionVar_0, %1)
      }

      def @collage_example_target_hook_nn_relu(%FunctionVar_03: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu") -> Tensor[(10, 10), float32] {
        %3 = fn (%FunctionVar_04: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_04)
        };
        %3(%FunctionVar_03)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        %4 = @collage_example_target_hook_nn_relu(%x);
        %5 = abs(%4);
        @collage_example_target_hook(%5)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_1_txt if byoc_max_depth == 1 else expected_3_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        }
    )
    run_collage(
        mod, targets, cost_estimator, expected_mod, tvm_max_depth=1, byoc_max_depth=byoc_max_depth
    )


@pytest.mark.parametrize("tvm_max_depth", [1, 2, 3])
@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_tvm_max_depth(mock_get_pattern_table, tvm_max_depth):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        nn.relu(%1)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txts = {
        1: """
          #[version = "0.0.5"]
          def @collage_example_target_hook(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook") -> Tensor[(10, 10), float32] {
            %0 = fn (%FunctionVar_03: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_03)
            };
            %1 = %0(%FunctionVar_0);
            %2 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_02)
            };
            %3 = %2(%1);
            %4 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_01)
            };
            %4(%3)
          }

          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            @collage_example_target_hook(%x)
          }
        """,
        2: """
          #[version = "0.0.5"]
          def @collage_example_target_hook_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu") -> Tensor[(10, 10), float32] {
            %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_01)
            };
            %0(%FunctionVar_0)
          }

          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            %1 = @collage_example_target_hook_nn_relu(%x);
            %2 = nn.relu(%1);
            nn.relu(%2)
          }
        """,
        3: """
          #[version = "0.0.5"]
          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            %0 = nn.relu(%x);
            %1 = nn.relu(%0);
            nn.relu(%1)
          }
        """,
    }
    expected_mod = tvm.parser.fromtext(expected_txts[tvm_max_depth])

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 100,
            "example_target_hook": 99,
        }
    )
    run_collage(
        mod, targets, cost_estimator, expected_mod, tvm_max_depth=tvm_max_depth, byoc_max_depth=1
    )


@pytest.mark.parametrize("byoc_max_depth", [1, 2, 3])
@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_byoc_max_depth(mock_get_pattern_table, byoc_max_depth):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        nn.relu(%1)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txts = {
        1: """
          #[version = "0.0.5"]
          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            %0 = nn.relu(%x);
            %1 = nn.relu(%0);
            nn.relu(%1)
          }
        """,
        2: """
          #[version = "0.0.5"]
          def @collage_example_target_hook_nn_relu_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu_nn_relu") -> Tensor[(10, 10), float32] {
            %0 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_02)
            };
            %1 = %0(%FunctionVar_0);
            %2 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_01)
            };
            %2(%1)
          }

          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            %3 = nn.relu(%x);
            @collage_example_target_hook_nn_relu_nn_relu(%3)
          }
        """,
        3: """
          #[version = "0.0.5"]
          def @collage_example_target_hook_nn_relu_nn_relu_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu_nn_relu_nn_relu") -> Tensor[(10, 10), float32] {
            %0 = fn (%FunctionVar_03: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_03)
            };
            %1 = %0(%FunctionVar_0);
            %2 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_02)
            };
            %3 = %2(%1);
            %4 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
              nn.relu(%FunctionVar_01)
            };
            %4(%3)
          }

          def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
            @collage_example_target_hook_nn_relu_nn_relu_nn_relu(%x)
          }
        """,
    }
    expected_mod = tvm.parser.fromtext(expected_txts[byoc_max_depth])

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 99,
            "example_target_hook": 100,
        }
    )
    run_collage(
        mod, targets, cost_estimator, expected_mod, tvm_max_depth=1, byoc_max_depth=byoc_max_depth
    )


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_partition_output_tuple(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        %2 = abs(%1);
        (%0, %1, %2)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook") -> (Tensor[(10, 10), float32], Tensor[(10, 10), float32]) {
        %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_01)
        };
        %1 = %0(%FunctionVar_0);
        %2 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_02)
        };
        %3 = %2(%1);
        (%1, %3)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> (Tensor[(10, 10), float32], Tensor[(10, 10), float32], Tensor[(10, 10), float32]) {
        %4 = @collage_example_target_hook(%x);
        %5 = %4.1;
        %6 = %4.0;
        %7 = abs(%5);
        (%6, %5, %7)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod, tvm_max_depth=2, byoc_max_depth=2)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_partition_intermediate_tuple(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        %2 = (%0, %1);
        concatenate(%2)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook(%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook") -> (Tensor[(10, 10), float32], Tensor[(10, 10), float32]) {
        %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_01)
        };
        %1 = %0(%FunctionVar_0);
        %2 = fn (%FunctionVar_02: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_02)
        };
        %3 = %2(%1);
        (%1, %3)
      }

      def @collage_example_target_hook_concatenate(%FunctionVar_03: (Tensor[(10, 10), float32], Tensor[(10, 10), float32]), Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_concatenate") -> Tensor[(20, 10), float32] {
        %4 = fn (%FunctionVar_04: (Tensor[(10, 10), float32], Tensor[(10, 10), float32]), Composite="concatenate") -> Tensor[(20, 10), float32] {
          concatenate(%FunctionVar_04)
        };
        %4(%FunctionVar_03)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(20, 10), float32] {
        %5 = @collage_example_target_hook(%x);
        %6 = %5.0;
        %7 = %5.1;
        %8 = (%6, %7);
        @collage_example_target_hook_concatenate(%8)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod, tvm_max_depth=3, byoc_max_depth=5)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_fusion_benefit(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        %2 = abs(%x);
        %3 = nn.relu(%2);
        %4 = add(%1, %3);
        %5 = nn.relu(%4);
        abs(%5)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook_nn_relu_nn_relu_nn_relu_add_nn_relu(%FunctionVar_0: Tensor[(10, 10), float32], %FunctionVar_1: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu_nn_relu_nn_relu_add_nn_relu") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_04: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_04)
        };
        %1 = %0(%FunctionVar_0);
        %2 = fn (%FunctionVar_03: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_03)
        };
        %3 = fn (%FunctionVar_05: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_05)
        };
        %4 = %2(%1);
        %5 = %3(%FunctionVar_1);
        %6 = fn (%FunctionVar_02: Tensor[(10, 10), float32], %FunctionVar_11: Tensor[(10, 10), float32], Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_02, %FunctionVar_11)
        };
        %7 = %6(%4, %5);
        %8 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_01)
        };
        %8(%7)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        %9 = abs(%x);
        %10 = @collage_example_target_hook_nn_relu_nn_relu_nn_relu_add_nn_relu(%x, %9);
        abs(%10)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 5,
            "example_target_hook": 6,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod, tvm_max_depth=1, byoc_max_depth=5)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_double_residual(mock_get_pattern_table):
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = abs(%0);
        %2 = add(%0, %1);
        add(%1, %2)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook_add_add(%FunctionVar_0: Tensor[(10, 10), float32], %FunctionVar_1: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_add_add") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_02: Tensor[(10, 10), float32], %FunctionVar_12: Tensor[(10, 10), float32], Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_02, %FunctionVar_12)
        };
        %1 = %0(%FunctionVar_1, %FunctionVar_0);
        %2 = fn (%FunctionVar_01: Tensor[(10, 10), float32], %FunctionVar_11: Tensor[(10, 10), float32], Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_01, %FunctionVar_11)
        };
        %2(%FunctionVar_0, %1)
      }

      def @collage_example_target_hook_nn_relu(%FunctionVar_03: Tensor[(10, 10), float32], Primitive=1, Compiler="example_target_hook", global_symbol="collage_example_target_hook_nn_relu") -> Tensor[(10, 10), float32] {
        %3 = fn (%FunctionVar_04: Tensor[(10, 10), float32], Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_04)
        };
        %3(%FunctionVar_03)
      }

      def @main(%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        %4 = @collage_example_target_hook_nn_relu(%x);
        %5 = abs(%4);
        @collage_example_target_hook_add_add(%5, %4)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]
    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        }
    )
    run_collage(mod, targets, cost_estimator, expected_mod, tvm_max_depth=4, byoc_max_depth=4)


@patch("tvm.relay.op.contrib.get_pattern_table", wraps=_mock_get_pattern_table)
def test_pruning_heuristic(mock_get_pattern_table):
    # In this example both the default TVM partition spec and the 'example_target_hook' partition
    # spec will yield the same set of candidates, and those candidates will include all 7
    # partitions of the four operators (ie 14 in total).
    #
    # However, the pruning heuristics will reduce those back to just two 'maximal' candidates
    # which have all four operators fused. We'll then just estimate those for the two targets.
    mod_txt = """
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = nn.relu(%x);
        %1 = nn.relu(%0);
        %2 = add(%0, %1);
        add(%1, %2)
      }
    """
    mod = tvm.parser.fromtext(mod_txt)

    expected_txt = """
      #[version = "0.0.5"]
      def @collage_example_target_hook_nn_relu_nn_relu_add_add(
        %FunctionVar_0: Tensor[(10, 10), float32],
        Primitive=1,
        Compiler="example_target_hook",
        global_symbol="collage_example_target_hook_nn_relu_nn_relu_add_add") -> Tensor[(10, 10), float32] {
        %0 = fn (%FunctionVar_03: Tensor[(10, 10), float32] , Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_03)
        };
        %1 = %0(%FunctionVar_0) ;
        %2 = fn (%FunctionVar_02: Tensor[(10, 10), float32] , Composite="relu") -> Tensor[(10, 10), float32] {
          nn.relu(%FunctionVar_02)
        };
        %3 = %2(%1);
        %4 = fn (%FunctionVar_04: Tensor[(10, 10), float32] , %FunctionVar_11: Tensor[(10, 10), float32] , Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_04, %FunctionVar_11)
        };
        %5 = %4(%1, %3);
        %6 = fn (%FunctionVar_01: Tensor[(10, 10), float32] , %FunctionVar_1: Tensor[(10, 10), float32] , Composite="add") -> Tensor[(10, 10), float32] {
          add(%FunctionVar_01, %FunctionVar_1)
        };
        %6(%3, %5)
      }

      def @main(%x: Tensor[(10, 10), float32] ) -> Tensor[(10, 10), float32] {
        @collage_example_target_hook_nn_relu_nn_relu_add_add(%x)
      }
    """
    expected_mod = tvm.parser.fromtext(expected_txt)

    targets = [
        tvm.target.Target("llvm"),
        tvm.target.Target("example_target_hook"),
    ]

    cost_estimator = MockCostEstimator(
        {
            "llvm": 2,
            "example_target_hook": 1,
        },
        # Limit the number of cost estimations to 2 to assert pruning did its job.
        max_estimates=2,
    )
    run_collage(mod, targets, cost_estimator, expected_mod, tvm_max_depth=4, byoc_max_depth=4)


if __name__ == "__main__":
    tvm.testing.main()
