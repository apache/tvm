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
    # elif isinstance(with_act, str) and (with_act == "nn.relu" or with_act == "nn.gelu" or with_act == "nn.hardswish"):
    #     return is_op(with_act)(gemm_out)
    # elif isinstance(with_act, str) and with_act == "gelu_asbl":
    #     power = is_op("power")(gemm_out, is_constant())
    #     multiply = is_op("multiply")(is_constant(), power)
    #     add = is_op("add")(gemm_out, multiply)
    #     multiply = is_op("multiply")(is_constant(), add)
    #     tanh = is_op("tanh")(multiply)
    #     add = is_op("add")(is_constant(), tanh)
    #     multiply = is_op("multiply")(is_constant(), add)
    #     multiply = is_op("multiply")(gemm_out, multiply)
    #     return multiply


@register_pattern_table("cutlass")
def pattern_table():
    dense_pat = ("cutlass.dense", make_gemm_pattern(False, None))
    # dense_bias_pat = ("cutlass.dense_bias",
    #                   make_gemm_pattern(True, None))
    # dense_bias_relu_pat = ("cutlass.dense_bias_relu",
    #                        make_gemm_pattern(True, "nn.relu"))
    # dense_bias_gelu_pat = ("cutlass.dense_bias_gelu",
    #                        make_gemm_pattern(True, "nn.gelu"))
    # dense_bias_hardswish_pat = ("cutlass.dense_bias_hardswish",
    #                        make_gemm_pattern(True, "nn.hardswish"))
    # dense_bias_gelu_asbl_pat = ("cutlass.dense_bias_gelu_asbl",
    #                        make_gemm_pattern(True, "gelu_asbl"))
    cutlass_patterns = [
        # dense_bias_gelu_asbl_pat,
        # dense_bias_gelu_pat,
        # dense_bias_hardswish_pat,
        # dense_bias_relu_pat,
        # dense_bias_pat,
        dense_pat,
    ]
    return cutlass_patterns
