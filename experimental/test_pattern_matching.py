import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    _DFPatternCallback,
    is_constant,
    is_op,
    rewrite,
    wildcard,
)


class BatchnormCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.var = wildcard()
        self.mean = wildcard()
        self.beta = wildcard()
        self.gamma = wildcard()
        self.eps = wildcard()

        self.pattern = (
            self.gamma * (self.x - self.mean) / is_op("sqrt")(self.var + self.eps) + self.beta
        )

    def callback(self, pre, post, node_map):
        import pdb

        pdb.set_trace()
        x = node_map[self.x][0]
        var = node_map[self.var][0]
        mean = node_map[self.mean][0]
        beta = node_map[self.beta][0]
        gamma = node_map[self.gamma][0]
        eps = node_map[self.eps][0]
        return relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=eps.data.asnumpy().item())[
            0
        ]


if __name__ == "__main__":
    # A graph of arithmetic operators that are functional equivalent to batch_norm.
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")
    BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )
