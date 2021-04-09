QParams = NamedTuple(
    "QParams", [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)]
)


class AffineQuantizationVarCreator:
    """Class which manages references to our qparams and can insert state."""

    def __init__(self):
        self.ref_count = 0
        self.qparams = []

    def get_qparams(self, name_hint: str, dtype: str = "int8") -> QParams:
        scale = relay.Var(f"{name_hint}.scale")
        zero_point = relay.Var(f"{name_hint}.zero_point")
        qparam = QParams(scale, zero_point, dtype)
        self.qparams.append(qparam)
        self.ref_count += 1
        return qparam
