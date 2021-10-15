import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def make_gemm_pattern(with_bias=True, with_act=None):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    gemm = is_op("nn.dense")(data, weight)
    if with_bias:
        add_or_bias_add = is_op("add") | is_op("nn.bias_add")
        gemm_out = add_or_bias_add(gemm, bias)
    else:
        gemm_out = gemm

    if with_act is None:
        return gemm_out
    elif isinstance(with_act, str) and with_act == "nn.relu":
        return is_op(with_act)(gemm_out)


@register_pattern_table("cutlass")
def pattern_table():
    dense_pat = ("cutlass.dense", make_gemm_pattern(False, None))
    dense_bias_pat = ("cutlass.dense_bias",
                      make_gemm_pattern(True, None))
    # dense_bias_relu_pat = ("cutlass.dense_bias_relu",
    #                        make_gemm_pattern(True, "nn.relu"))
    # dense_bias_gelu_pat = ("cutlass.dense_bias_gelu",
    #                        make_gemm_pattern(True, "nn.gelu"))
    cutlass_patterns = [
        # dense_bias_gelu_pat,
        # dense_bias_relu_pat,
        dense_bias_pat,
        dense_pat,
    ]
    return cutlass_patterns
