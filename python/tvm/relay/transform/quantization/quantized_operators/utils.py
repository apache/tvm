from typing import NamedTuple, Optional

import numpy as np
import tvm
from tvm import relay

QParams = NamedTuple(
    "QParams", [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)]
)


class AffineQuantizationVarCreator:
    """Class which manages references to our qparams and can insert state."""

    def __init__(self):
        self.ref_count = 0
        self.qparams = []

    def get_qparams(self, name_hint: str, dtype: str = "int8") -> QParams:
        scale = relay.var(f"{name_hint}.scale", type_annotation="float32")
        zero_point = relay.var(f"{name_hint}.zero_point", type_annotation="int8")
        qparam = QParams(scale, zero_point, dtype)
        self.qparams.append(qparam)
        self.ref_count += 1
        return qparam


def get_quantization_parameters(
    data: np.array,
    signed: bool,
    nbits: int,
    per_channel: Optional[int] = None,
    symmetric: bool = False,
) -> QParams:
    if per_channel is not None:
        raise ValueError("We do not support per channel quantization right now!")

    if not signed or nbits != 8:
        raise ValueError("Only support int8 right now lol.")

    data = data.flatten()
    rmin, rmax = data.min(), data.max()

    # Always need to represent 0!
    if rmax > 0:
        rmin = min(rmin, 0)
    else:
        rmax = max(rmax, 0)

    scale = (rmax - rmin) / (2 ** nbits - 1)
    if symmetric:
        zero_point = 0
    else:
        # This zero point will be based on an unsigned version of nbits, we'll adjust later
        zero_point = abs(rmin // scale)

        # Readjust scale to max sure we can represent both the min and max still
        rmax_scale = rmax / (2 ** nbits - 1 - zero_point)
        rmin_scale = abs(rmin / zero_point)
        scale = max(rmax_scale, rmin_scale)
    if signed:
        zero_point -= 2 ** (nbits - 1)

    return QParams(
        relay.const(np.float32(scale)),
        relay.const(np.int8(zero_point)),
        "int8",
    )


if __name__ == "__main__":
    example_input = np.random.uniform(-5, 10, size=(5, 5)).astype("float32")
    params = get_quantization_parameters(example_input, True, 8)
    data = relay.var("data")

    print("**Input:")
    print(example_input)
    print(params)
    print()

    q_data = relay.qnn.op.simulated_quantize(data, params.scale_factor, params.zero_point)
    q_data_func = relay.Function([data], q_data)
    mod = tvm.ir.IRModule.from_expr(q_data_func)
    intrp = relay.create_executor(kind="debug", mod=mod)

    print("**Q_Input:")
    print(intrp.evaluate(q_data_func)(example_input).asnumpy())
    print()

    dq_q_data = relay.qnn.op.simulated_dequantize(q_data, params.scale_factor, params.zero_point)
    dq_q_data_func = relay.Function([data], dq_q_data)
    mod = tvm.ir.IRModule.from_expr(dq_q_data_func)
    intrp = relay.create_executor(kind="debug", mod=mod)

    print("**FQ_Input:")
    print(intrp.evaluate(dq_q_data_func)(example_input).asnumpy())
