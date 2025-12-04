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
                # (64, 32): "FloatToPosit32es2",
                # (64, 16): "FloatToPosit16es2",
                # (64, 8): "FloatToPosit8es2",
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
                # (32, 64): "Posit32es2ToFloat",
                # (16, 64): "Posit16es2ToFloat",
                # (8, 64): "Posit8es2ToFloat",
            }
        ),
        "Cast",
        "llvm",
        "posites2",
        "float",
    )

    # posites2 -> uint: includes ToBool and identity casts
    register_op(
        create_lower_func(
            {
                (32, 1): "Posit32es2ToBool",
                (16, 1): "Posit16es2ToBool",
                (8, 1):  "Posit8es2ToBool",
                (32, 8): "Posit32es2ToBool",
                (16, 8): "Posit16es2ToBool",
                (8, 8):  "Posit8es2ToBool",
                (32, 32): "Posit32es2ToUint32",  # posit32 -> uint32: identity (same bits)
                (16, 16): "Posit16es2ToUint16",  # posit16 -> uint16: identity
                (8, 8): "Posit8es2ToUint8",      # posit8 -> uint8: identity (already handled above but explicit here)
            }
        ),
        "Cast",
        "llvm",
        "posites2", 
        "uint",
    )

    # uint -> posites2: includes bool conversion and identity casts
    register_op(
        create_lower_func(
            {
                (1, 32): "BoolToPosit32es2",
                (1, 16): "BoolToPosit16es2",
                (1, 8):  "BoolToPosit8es2",
                (32, 32): "Uint32ToPosit32es2",  # uint32 -> posit32: identity (same bits)
                (16, 16): "Uint16ToPosit16es2",  # uint16 -> posit16: identity
                (8, 8): "Uint8ToPosit8es2",      # uint8 -> posit8: identity
            }
        ),
        "Cast",
        "llvm",
        "uint", 
        "posites2",
    )
    
    register_op(
        create_lower_func(
            {
                (64, 32): "IntToPosit32es2",
                (64, 16): "IntToPosit16es2",
                # (64, 8): "IntToPosit8es2",
                # (32, 32): "IntToPosit32es2",
                # (32, 16): "IntToPosit16es2",
                # (32, 8): "IntToPosit8es2",
                (16, 16): "IntToPosit16es2",
                # (16, 8): "IntToPosit8es2",
                (8, 8): "IntToPosit8es2",
            }
        ),
        "Cast",
        "llvm",
        "int",
        "posites2",
    )

    register_op(
        create_lower_func(
            {
                # (32, 64): "Posit32es2ToInt",
                (32, 32): "Posit32es2ToInt",
                # (32, 16): "Posit32es2ToInt",
                # (32, 8): "Posit32es2ToInt",
                # (16, 64): "Posit16es2ToInt",
                # (16, 32): "Posit16es2ToInt",
                (16, 16): "Posit16es2ToInt",
                # (16, 8): "Posit16es2ToInt",
                # (8, 64): "Posit8es2ToInt",
                # (8, 32): "Posit8es2ToInt",
                # (8, 16): "Posit8es2ToInt",
                (8, 8): "Posit8es2ToInt",
            }
        ),
        "Cast",
        "llvm",
        "posites2",
        "int",
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
        create_lower_func({32: "Posit32es2Min", 16: "Posit16es2Min", 8: "Posit8es2Min"}),
        "Min",
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
    register_op(
        create_lower_func({32: "Posit32es2Pow", 16: "Posit16es2Pow", 8: "Posit8es2Pow"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.pow",
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
    register_op(
        create_lower_func({32: "Posit32es2Cos", 16: "Posit16es2Cos", 8: "Posit8es2Cos"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.cos",
    )
    register_op(
        create_lower_func({32: "Posit32es2Sin", 16: "Posit16es2Sin", 8: "Posit8es2Sin"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.sin",
    )
    register_op(
        create_lower_func({32: "Posit32es2Tan", 16: "Posit16es2Tan", 8: "Posit8es2Tan"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.tan",
    )
    register_op(
        create_lower_func({32: "Posit32es2Erf", 16: "Posit16es2Erf", 8: "Posit8es2Erf"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.erf",
    )
    register_op(
        create_lower_func({32: "Posit32es2Softmax", 16: "Posit16es2Softmax", 8: "Posit8es2Softmax"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.softmax",
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


