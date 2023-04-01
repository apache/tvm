"""Elementwise operators, including unary and binary broadcast."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import BoolTensor, FloatTensor, IntTensor, Tensor


def _unary(name, a_type, doc):
    class _Unary:
        a = a_type
        ret = Tensor

    _Unary.__doc__ = doc
    register_op(
        name,
        category="elementwise",
        te_func="topi.array_api." + name,
    )(_Unary)


def _binary(name, a_type, b_type, doc):
    class _Binary:
        a = a_type
        b = b_type
        ret = Tensor

    _Binary.__doc__ = doc
    register_op(
        name,
        category="elementwise",
        te_func="topi.array_api." + name,
    )(_Binary)


# ##### Category 1. Basic arithmetic operators
#
# _binary("add", Tensor, Tensor, "Elementwise add")
# _binary("subtract", Tensor, Tensor, "Elementwise subtract")
# _binary("multiply", Tensor, Tensor, "Elementwise multiply")
# _binary("divide", Tensor, Tensor, "Elementwise divide")
# _binary("floor_divide", Tensor, Tensor, "Elementwise floor divide")
# _binary("remainder", Tensor, Tensor, "Elementwise remainder")
# _binary("pow", Tensor, Tensor, "Elementwise pow")
#
# ##### Category 2. Trigonometric functions
#
_unary("acos", FloatTensor, "Elementwise acos")
# _unary("acosh", FloatTensor, "Elementwise acosh")
# _unary("asin", FloatTensor, "Elementwise asin")
# _unary("asinh", FloatTensor, "Elementwise asinh")
# _unary("atan", FloatTensor, "Elementwise atan")
# _unary("atanh", FloatTensor, "Elementwise atanh")
# _unary("cos", FloatTensor, "Elementwise cos")
# _unary("cosh", FloatTensor, "Elementwise cosh")
# _unary("sin", FloatTensor, "Elementwise sin")
# _unary("sinh", FloatTensor, "Elementwise sinh")
# _unary("tan", FloatTensor, "Elementwise tan")
# _unary("tanh", FloatTensor, "Elementwise tanh")
# _binary("atan2", FloatTensor, FloatTensor, "Elementwise atan2")
#
# ##### Category 3. Exp/log/square operators
#
# _unary("exp", FloatTensor, "Elementwise exp")
# # _unary("expm1", FloatTensor, "Elementwise expm1")  # TODO(@junrushao): add
# _unary("log", FloatTensor, "Elementwise log")
# _unary("log1p", FloatTensor, "Elementwise log1p")
# _unary("log2", FloatTensor, "Elementwise log2")
# _unary("log10", FloatTensor, "Elementwise log10")
# # _unary("square", FloatTensor, "Elementwise square")  # TODO(@junrushao): add
# _unary("sqrt", FloatTensor, "Elementwise sqrt")
#
# ##### Category 4. Rounding/Sign operators
#
# _unary("abs", FloatTensor, "Elementwise abs")
# _unary("ceil", FloatTensor, "Elementwise ceil")
# _unary("floor", FloatTensor, "Elementwise floor")
# _unary("trunc", FloatTensor, "Elementwise trunc")
# _unary("round", FloatTensor, "Elementwise round")
# # _unary("sign", FloatTensor, "Elementwise sign")  # TODO(@junrushao): add intrin: sign
# _unary("positive", FloatTensor, "Elementwise positive")
# _unary("negative", FloatTensor, "Elementwise negative")
#
#
# ##### Category 5. Comparison operators
#
# _binary("equal", Tensor, Tensor, "Elementwise equal")
# _binary("not_equal", Tensor, Tensor, "Elementwise not equal")
# _binary("greater", Tensor, Tensor, "Elementwise greater")
# _binary("greater_equal", Tensor, Tensor, "Elementwise greater equal")
# _binary("less", Tensor, Tensor, "Elementwise less")
# _binary("less_equal", Tensor, Tensor, "Elementwise less equal")
#
#
# ##### Category 6. Bitwise operators
#
# _binary("bitwise_and", IntTensor, IntTensor, "Elementwise bitwise and")
# _binary("bitwise_or", IntTensor, IntTensor, "Elementwise bitwise or")
# _binary("bitwise_xor", IntTensor, IntTensor, "Elementwise bitwise xor")
# _unary("bitwise_invert", IntTensor, "Elementwise bitwise invert")
# _binary("bitwise_left_shift", IntTensor, IntTensor, "Elementwise bitwise left shift")
# _binary("bitwise_right_shift", IntTensor, IntTensor, "Elementwise bitwise right shift")
#
#
# ##### Category 7. Logical operators
#
# _binary("logical_and", BoolTensor, BoolTensor, "Elementwise logical and")
# _binary("logical_or", BoolTensor, BoolTensor, "Elementwise logical or")
# _unary("logical_not", BoolTensor, "Elementwise logical not")
# # _binary("logical_xor", BoolTensor, BoolTensor, "Elementwise logical xor")  # TODO(@junrushao): add
#
# ##### Category 8. Special value checking for floating point numbers
#
# _unary("isfinite", FloatTensor, "Elementwise isfinite")
# _unary("isinf", FloatTensor, "Elementwise isinf")
# _unary("isnan", FloatTensor, "Elementwise isnan")
