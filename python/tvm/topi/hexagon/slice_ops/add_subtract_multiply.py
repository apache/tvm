# pylint: disable=invalid-name

from tvm import te
from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn


def add_broadcast_compute(A, B):
    return topi.add(A, B)


def subtract_broadcast_compute(A, B):
    return topi.subtract(A, B)


def multiply_broadcast_compute(A, B):
    return topi.multiply(A, B)


def get_layout(layout):
    layout += "-2d"
    return get_layout_transform_fn(layout)


def STIR_broadcast_schedule(
    M, A, B, output_layout: str, input_A_layout: str, input_B_layout: str, op_name: str
):
    func = te.create_prim_func([A, B, M])

    s = tir.Schedule(func)

    block_dict = {"add": "T_add", "subtract": "T_subtract", "multiply": "T_multiply"}

    block = s.get_block(block_dict[op_name])

    if input_A_layout == "nhwc-8h2w32c2w":
        input_A_transformed_layout = get_layout(input_A_layout)
        s.transform_layout(block, buffer=("read", 0), index_map=input_A_transformed_layout)

    if input_B_layout == "nhwc-8h2w32c2w":
        input_B_transformed_layout = get_layout(input_B_layout)
        s.transform_layout(block, buffer=("read", 1), index_map=input_B_transformed_layout)

    output_transformed_layout = get_layout(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    n, h, w, c = s.get_loops(block)

    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 4])
    co, ci = s.split(c, [None, 32])
    wio, wii = s.split(wi, [None, 2])

    s.reorder(n, ho, wo, co, hi, wio, ci, wii)

    fused = s.fuse(ci, wii)
    s.vectorize(fused)

    return s
