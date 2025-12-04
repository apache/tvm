import os
import sys
from tvm import IRModule, tir
import tvm
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax import transform
import onnx
import time
from tvm.relax.frontend.change_datatype import ChangeDatatype

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from BYODT.register import _posit_registered

_posit_registered()

project_root = os.path.abspath(sys.path[0])
os.chdir(project_root)

pre_tuning_pipeline = tvm.transform.Sequential(
    [
        transform.DecomposeOpsForInference(),
        transform.CanonicalizeBindings(),
        tvm.relax.get_pipeline("zero"),
        transform.AttachAttrLayoutFreeBuffers()
    ]
)
post_tuning_pipeline = tvm.transform.Sequential(
    [
        transform.SplitLayoutRewritePreproc(),
        transform.LiftTransformParams(),
        transform.FoldConstant(),
    ]
)
# Helper function to retrieve all blocks in a TIR function
def get_all_blocks(sch, func):
    blocks = []
    # Define a visitor to collect all blocks
    def visit_block(stmt):
        if isinstance(stmt, tir.Block):
            blocks.append(stmt)
    tir.stmt_functor.post_order_visit(func.body, visit_block)
    return blocks

# Function to apply automatic optimization to a given TIR function
def auto_optimize_func(sch, func, tile_sizes=(8, 16, 32, 16), use_vectorize=False):
    def custom_sort_key(item):
        index, value = item
        if isinstance(value, str):
            return (0, value)
        elif isinstance(value, int):
            return (1, -value) 
        return (2, index)

    # Collect all block names in the function
    blocks = get_all_blocks(sch, func)
    # Apply scheduling transformations to each block
    for block in blocks:
        block_name = block.name_hint
        # Get the target block
        blockRV = sch.get_block(block_name)
        # Get the loops of the block
        loops = sch.get_loops(blockRV)
        if len(loops) == 0:
            continue

        iter_vars = []
        for index, iter_var in enumerate(block.iter_vars):
            if iter_var.iter_type == iter_var.CommReduce:
                continue
            if type(iter_var.dom.extent) is tir.expr.IntImm:
                value = iter_var.dom.extent.value
            else:
                value = 'none'
            iter_vars.append(value)
        if len(iter_vars) > 0:
            iter_vars = [(index, value) for index, value in enumerate(iter_vars)]
            print(iter_vars)
            sorted_list = sorted(iter_vars, key=custom_sort_key)
            # print("block:", block_name)
            # print("  #loops:", len(loops))
            # print("  iter_vars:", iter_vars)
            # print("  sorted_list:", sorted_list)
            # Check if there are enough loops to tile
            #if 'matmul' in block_name:

            reorder_list = []
            for i in range(len(sorted_list)):
                reorder_list.append(loops[sorted_list[i][0]])
            sch.reorder(*reorder_list)

            for i in range(0, len(sorted_list)):
                sch.parallel(loops[sorted_list[i][0]])
                sorted_list[i] = (-1, sorted_list[i][1])
                break
            
            # Use vectorize for float types
            if use_vectorize:
                for i in range(0, len(sorted_list)):
                    if (
                        sorted_list[i][0] != -1
                        and isinstance(sorted_list[i][1], int)
                        and sorted_list[i][1] > 1
                        and sorted_list[i][1] <= 256
                    ):
                        sch.vectorize(loops[sorted_list[i][0]])
                        sorted_list[i] = (-1, sorted_list[i][1])
                        break

            for i in range(0, len(sorted_list)):
                if (
                    sorted_list[i][0] != -1
                    and isinstance(sorted_list[i][1], int)
                    and sorted_list[i][1] <= 128
                ):
                    sch.unroll(loops[sorted_list[i][0]])
                    sorted_list[i] = (-1, sorted_list[i][1])
                    break


    #print(sch.mod.script())
    # Return the modified function from the schedule
    return sch.mod["main"]

# Function to optimize all TIR functions in the IR module
def optimize_ir_module(ir_module, use_vectorize=False):
    optimized_module = IRModule()

    # Iterate over each function in the IR module
    for name, func in ir_module.functions.items():
        if isinstance(func, tir.PrimFunc):  # Only apply to TIR functions
            # Create a schedule and apply the optimizations
            print(f"Operator: {name}")
            sch = tir.Schedule(func)
            optimized_func = auto_optimize_func(sch, func, use_vectorize=use_vectorize)
            optimized_module[name] = optimized_func
        else:
            optimized_module[name] = func

    return optimized_module

def hand_sch(mod):
    mod = pre_tuning_pipeline(mod)
    mod = post_tuning_pipeline(mod)
    mod = optimize_ir_module(mod)
    return mod

def compile_model(onnx_path, dtype_converter=None, target_bits=None, use_vectorize=False):
    """Load ONNX model and apply transformations
    
    Args:
        onnx_path: Path to ONNX model
        dtype_converter: Optional function to convert data types
        target_bits: Optional target bit width (16 or 32) for weight compression
        use_vectorize: Whether to use vectorize optimization (for float types)
    """
    mod = onnx.load_model(onnx_path)
    mod = from_onnx(mod, {"input_features": (1, 80, 3000)})
    
    if dtype_converter:
        new_main = dtype_converter(mod)
        mod = tvm.IRModule({"main": new_main})
    
    mod = tvm.relax.transform.LegalizeOps()(mod)
    
    # Apply constant folding and lifting to compress weights
    # mod = tvm.relax.transform.LiftTransformParams()(mod)
    # mod = tvm.relax.transform.FoldConstant()(mod)

    mod = optimize_ir_module(mod, use_vectorize=use_vectorize)
    
    return mod

# Configuration
target = tvm.target.Target('llvm --num-cores=16 -mattr=+avx2,+sse4.2,+sse4.1')
base_path = "/home/wang/venv/HPCLab/MS_LLM/models/whisper-tiny/onnx"

# Compile FP32
print("Compiling FP32 model...")
mod_fp32 = compile_model(f"{base_path}/encoder_model.onnx", use_vectorize=True)
start = time.time()
lib = tvm.relax.build(mod_fp32, target=target)
fp32_build_time = time.time() - start
print(f"FP32 build time: {fp32_build_time:.2f} seconds")
lib.export_library("./model/encoder_fp32.so")

# Compile FP16
print("Compiling FP16 model...")
mod_fp16 = compile_model(f"{base_path}/encoder_model_fp16.onnx", use_vectorize=True)
start = time.time()
lib_fp16 = tvm.relax.build(mod_fp16, target=target)
fp16_build_time = time.time() - start
print(f"FP16 build time: {fp16_build_time:.2f} seconds")
lib_fp16.export_library("./model/encoder_fp16.so")

# Compile Posit models
# print("Compiling Posit32 model...")
# mod_base = onnx.load_model(f"{base_path}/encoder_model.onnx")
# mod_base = from_onnx(mod_base, {"input_features": (1, 80, 3000)})
# start_time = time.time()
# dtype_mutator = ChangeDatatype("float32", "custom[posites2]32", mod_base)
# mod_posit32 = compile_model(f"{base_path}/encoder_model.onnx", 
#                              lambda m: dtype_mutator.visit_expr(m["main"]))
# lib_posit32 = tvm.relax.build(mod_posit32, target=target)
# posit32_build_time = time.time() - start_time
# print(f"Posit32 build time: {posit32_build_time:.2f} seconds")
# start_time = time.time()
# lib_posit32.export_library("./model/encoder_posit32es2.so")
# posit32_export_time = time.time() - start_time
# print(f"Posit32 export time: {posit32_export_time:.2f} seconds")

print("Done")
