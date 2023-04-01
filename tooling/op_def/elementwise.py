"""Elementwise operators, including unary and binary broadcast."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import BoolTensor, FloatTensor, IntTensor, Tensor


def _unary(
    name,
    a_type,
    doc,
    out_dtype=None,
):
    class _Unary:
        a = a_type
        ret = Tensor

    _Unary.__doc__ = doc
    if out_dtype is None:
        out_dtype = "Void()"

    name_hint = "tir_" + name
    register_op(
        name,
        category="elementwise",
        te_func="topi.array_api." + name,
        sinfo=f'UnarySInfo("topi.array_api.{name}", runtime::DataType::{out_dtype})',
        legalization=f'UnaryLegalize("topi.array_api.{name}", "{name_hint}")',
        attrs=[
            ("FRelaxInferLayout", "InferLayoutUnaryEwise"),
            ("TMixedPrecisionPolicy", "MixedPrecisionPolicyKind::kFollow"),
        ],
    )(_Unary)


def _binary(  # pylint: disable=too-many-arguments
    name,
    a_type,
    b_type,
    doc,
    out_dtype=None,
    name_hint=None,
):
    class _Binary:
        a = a_type()
        b = b_type()
        ret = Tensor

    _Binary.__doc__ = doc
    if out_dtype is None:
        out_dtype = "Void()"
    if name_hint is None:
        name_hint = name
    _Binary.a.maybe_scalar = True
    _Binary.b.maybe_scalar = True
    register_op(
        name,
        category="elementwise",
        te_func="topi.array_api." + name,
        sinfo=f'BinarySInfo("topi.array_api.{name}", runtime::DataType::{out_dtype})',
        legalization=f'BinaryLegalize("topi.array_api.{name}", "{name_hint}")',
        attrs=[
            ("FRelaxInferLayout", "InferLayoutBinaryEwise"),
            ("TMixedPrecisionPolicy", "MixedPrecisionPolicyKind::kFollow"),
        ],
    )(_Binary)


# ##### Category 1. Basic arithmetic operators

_binary("add", Tensor, Tensor, "Elementwise add")
_binary("subtract", Tensor, Tensor, "Elementwise subtract")
_binary("multiply", Tensor, Tensor, "Elementwise multiply")
_binary("divide", Tensor, Tensor, "Elementwise divide")  # TODO: output dtype?
_binary("floor_divide", Tensor, Tensor, "Elementwise floor divide")
_binary("remainder", Tensor, Tensor, "Elementwise remainder")
_binary("pow", Tensor, Tensor, "Elementwise pow")  # TODO: output dtype?
_binary("power", Tensor, Tensor, "Elementwise pow")

# ##### Category 2. Trigonometric functions

_unary("acos", FloatTensor, "Elementwise acos")
_unary("acosh", FloatTensor, "Elementwise acosh")
_unary("asin", FloatTensor, "Elementwise asin")
_unary("asinh", FloatTensor, "Elementwise asinh")
_unary("atan", FloatTensor, "Elementwise atan")
_unary("atanh", FloatTensor, "Elementwise atanh")
_unary("cos", FloatTensor, "Elementwise cos")
_unary("cosh", FloatTensor, "Elementwise cosh")
_unary("sin", FloatTensor, "Elementwise sin")
_unary("sinh", FloatTensor, "Elementwise sinh")
_unary("tan", FloatTensor, "Elementwise tan")
_unary("tanh", FloatTensor, "Elementwise tanh")
_binary("atan2", FloatTensor, FloatTensor, "Elementwise atan2")

# ##### Category 3. Exp/log/square operators

_unary("exp", FloatTensor, "Elementwise exp")
# _unary("expm1", FloatTensor, "Elementwise expm1")  # TODO
_unary("log", FloatTensor, "Elementwise log")
_unary("log1p", FloatTensor, "Elementwise log1p")
_unary("log2", FloatTensor, "Elementwise log2")
_unary("log10", FloatTensor, "Elementwise log10")
_unary("square", Tensor, "Elementwise square")  # TODO(@junrushao): do we need float?
_unary("sqrt", FloatTensor, "Elementwise sqrt")

# ##### Category 4. Rounding/Sign operators

_unary("abs", Tensor, "Elementwise abs")
_unary("ceil", Tensor, "Elementwise ceil")  # TODO(@junrushao): do we need float?
_unary("floor", Tensor, "Elementwise floor")  # TODO(@junrushao): do we need float?
_unary("trunc", FloatTensor, "Elementwise trunc")
_unary("round", Tensor, "Elementwise round")  # TODO(@junrushao): do we need float?
# _unary("sign", FloatTensor, "Elementwise sign")  # TODO(@junrushao): add intrin: sign
_unary("positive", Tensor, "Elementwise positive")
_unary("negative", Tensor, "Elementwise negative")


# ##### Category 5. Comparison operators

_binary("equal", Tensor, Tensor, "Elementwise equal", out_dtype="Bool()")
_binary("not_equal", Tensor, Tensor, "Elementwise not equal", out_dtype="Bool()")
_binary("greater", Tensor, Tensor, "Elementwise greater", out_dtype="Bool()")
_binary("greater_equal", Tensor, Tensor, "Elementwise greater equal", out_dtype="Bool()")
_binary("less", Tensor, Tensor, "Elementwise less", out_dtype="Bool()")
_binary("less_equal", Tensor, Tensor, "Elementwise less equal", out_dtype="Bool()")


# ##### Category 6. Bitwise operators

_binary("bitwise_and", IntTensor, IntTensor, "Elementwise bitwise and")
_binary("bitwise_or", IntTensor, IntTensor, "Elementwise bitwise or")
_binary("bitwise_xor", IntTensor, IntTensor, "Elementwise bitwise xor")
_unary("bitwise_invert", IntTensor, "Elementwise bitwise invert")
_binary("bitwise_left_shift", IntTensor, IntTensor, "Elementwise bitwise left shift")
_binary("bitwise_right_shift", IntTensor, IntTensor, "Elementwise bitwise right shift")


# ##### Category 7. Logical operators

_binary("logical_and", BoolTensor, BoolTensor, "Elementwise logical and")
_binary("logical_or", BoolTensor, BoolTensor, "Elementwise logical or")
_unary("logical_not", BoolTensor, "Elementwise logical not")
# # _binary("logical_xor", BoolTensor, BoolTensor, "Elementwise logical xor")  # TODO(@junrushao): add

# ##### Category 8. Special value checking for floating point numbers

_unary("isfinite", FloatTensor, "Elementwise isfinite", out_dtype="Bool()")
_unary("isinf", FloatTensor, "Elementwise isinf", out_dtype="Bool()")
_unary("isnan", FloatTensor, "Elementwise isnan", out_dtype="Bool()")
