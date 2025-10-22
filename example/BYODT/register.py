from tvm.target.datatype import (
    create_lower_func,
    create_min_lower_func,
    lower_call_pure_extern,
    lower_ite,
    register,
    register_min_func,
    register_op,
)

_POSIT_REGISTERED = False

def _posit_registered():
    global _POSIT_REGISTERED
    if _POSIT_REGISTERED:
        return

    register("posites2", 132)

    register_op(
        create_lower_func(
            {
                (32, 32): "FloatToPosit32es2",
                (32, 16): "FloatToPosit16es2",
                (32, 8): "FloatToPosit8es2",
            }
        ),
        "Cast",
        "llvm",
        "float",
        "posites2",
    )
    register_op(
        create_lower_func(
            {
                (32, 32): "Posit32es2ToFloat",
                (16, 32): "Posit16es2ToFloat",
                (8, 32): "Posit8es2ToFloat",
            }
        ),
        "Cast",
        "llvm",
        "posites2",
        "float",
    )
    register_op(
        create_lower_func({32: "Posit32es2Add", 16: "Posit16es2Add", 8: "Posit8es2Add"}),
        "Add",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Sub", 16: "Posit16es2Sub", 8: "Posit8es2Sub"}),
        "Sub",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func(
            {32: "FloatToPosit32es2", 16: "FloatToPosit16es2", 8: "FloatToPosit8es2"}
        ),
        "FloatImm",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Mul", 16: "Posit16es2Mul", 8: "Posit8es2Mul"}),
        "Mul",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Div", 16: "Posit16es2Div", 8: "Posit8es2Div"}),
        "Div",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Max", 16: "Posit16es2Max", 8: "Posit8es2Max"}),
        "Max",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Sqrt", 16: "Posit16es2Sqrt", 8: "Posit8es2Sqrt"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.sqrt",
    )
    register_op(lower_ite, "Call", "llvm", "posites2", intrinsic_name="tir.if_then_else")
    register_op(
        lower_call_pure_extern, "Call", "llvm", "posites2", intrinsic_name="tir.call_pure_extern"
    )
    register_op(
        create_lower_func({32: "Posit32es2Exp", 16: "Posit16es2Exp", 8: "Posit8es2Exp"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.exp",
    )
    register_op(
        create_lower_func({32: "Posit32es2Log", 16: "Posit16es2Log", 8: "Posit8es2Log"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.log",
    )
    register_op(
        create_lower_func(
            {32: "Posit32es2Sigmoid", 16: "Posit16es2Sigmoid", 8: "Posit8es2Sigmoid"}
        ),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.sigmoid",
    )
    register_op(
        create_lower_func({32: "Posit32es2Tanh", 16: "Posit16es2Tanh", 8: "Posit8es2Tanh"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.tanh",
    )

    register_min_func(
        create_min_lower_func(
            {32: "Posit32es2Min", 16: "Posit16es2Min", 8: "Posit8es2Min"}, "posites2"
        ),
        "posites2",
    )

    _POSIT_REGISTERED = True

_posit_registered()
__all__ = ["_posit_registered"]


