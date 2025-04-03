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
from tvm.relax.frontend import nn

when_to_apply_replacement = tvm.testing.parameter(
    "relax",
    "slm",
)


def test_replace_implementation(when_to_apply_replacement):
    class FeedForward(nn.Module):
        """The base implementation to be replaced"""

        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                bias=False,
            )
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, hidden_states: nn.Tensor):
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            return self.down_proj(nn.op.silu(gate) * up)

    class OptimizedFeedForward(FeedForward):
        """The optimized implementation"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @property
        def gate_up_proj(self):
            hidden_size = self.gate_proj.in_features
            intermediate_size = self.gate_proj.out_features

            gate_up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=2 * intermediate_size,
                bias=False,
            )
            gate_up_proj.weight = nn.op.concat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            return gate_up_proj

        def forward(self, hidden_states: nn.Tensor):
            concat_x1_x2 = self.gate_up_proj(hidden_states)
            gate, up = nn.op.split(concat_x1_x2, 2, axis=-1)
            return self.down_proj(nn.op.silu(gate) * up)

    class DecoderLayer(nn.Module):
        """A Module that internally uses the base implementation"""

        def __init__(self, hidden_side: int, intermediate_size: int):
            self.self_attn = nn.Identity()  # For sake of testing
            self.mlp = FeedForward(hidden_size, intermediate_size)
            self.input_layernorm = nn.RMSNorm(hidden_size, axes=-1, bias=False)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, axes=-1, bias=False)

        def forward(self, hidden_states: nn.Tensor):
            hidden_states += self.self_attn(self.input_layernorm(hidden_states))
            hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states

    class ExpectedDecoderLayer(nn.Module):
        """A Module that internally uses the optimized implementation

        This class is for testing purposes.  After injecting the
        optimized implementation, we should produce equivalent
        end-to-end SLM/Relax models as the hand-written version.
        """

        def __init__(self, hidden_side: int, intermediate_size: int):
            self.self_attn = nn.Identity()  # For sake of testing
            self.mlp = OptimizedFeedForward(hidden_size, intermediate_size)
            self.input_layernorm = nn.RMSNorm(hidden_size, axes=-1, bias=False)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, axes=-1, bias=False)

        def forward(self, hidden_states: nn.Tensor):
            hidden_states += self.self_attn(self.input_layernorm(hidden_states))
            hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states

    batch_size = 16
    hidden_size = 4096
    intermediate_size = 11008
    dtype = "float32"

    slm_model = DecoderLayer(intermediate_size, hidden_size)

    mutator = nn.replace_implementation(OptimizedFeedForward)

    if when_to_apply_replacement == "slm":
        slm_model = mutator.visit("", slm_model)

    model_expected = ExpectedDecoderLayer(intermediate_size, hidden_size)

    spec = {
        "forward": {"hidden_states": nn.spec.Tensor([batch_size, hidden_size], dtype)},
    }

    relax_expected = model_expected.export_tvm(spec)[0]
    assert tvm.relax.analysis.well_formed(relax_expected)

    relax_model = slm_model.export_tvm(spec)[0]
    assert tvm.relax.analysis.well_formed(relax_model)

    if when_to_apply_replacement == "relax":
        transform = mutator.as_relax_transform()
        relax_model = transform(relax_model)

    normalize = tvm.ir.transform.Sequential(
        [
            # Normalize the IRModule by apply a topological sort within each
            # dataflow block.  Otherwise, equivalent replacements performed at
            # a different step of optimization can result in a different order
            # of intermediates.
            tvm.relax.transform.TopologicalSort(order="depth-first", direction="from-outputs"),
            # The SLM exporter produces a trivial `var = dataflow_var`
            # binding for the output, which should be removed before
            # validating the output.
            tvm.relax.transform.CanonicalizeBindings(),
        ]
    )
    relax_model = normalize(relax_model)
    relax_expected = normalize(relax_expected)

    tvm.ir.assert_structural_equal(relax_model, relax_expected)


if __name__ == "__main__":
    tvm.testing.main()
