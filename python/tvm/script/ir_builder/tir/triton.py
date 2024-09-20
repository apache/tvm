import triton
from triton.runtime.jit import type_canonicalisation_dict
from tvm import tir
from tvm.topi.utils import get_const_int
from tvm.runtime import load_module, Module
import tempfile
import json
from tvm._ffi.libinfo import __version__

from typing import Tuple, List, Union, Any


def _generate_triton_kernel(
    func, grid, *args, **kwargs
) -> Union[triton.compiler.CompiledKernel, List[tir.PrimExpr], List[str]]:
    """Deduce the kernel signature and generate the Triton kernel"""

    kernel_params = func.params
    assert len(kernel_params) == len(
        args
    ), f"Number of arguments does not match, expected {len(kernel_params)}, got {len(args)}"

    # Step 1: Generate the Triton kernel
    signature = {}
    constants = {}
    # Arguments to invoke the kernel
    kernel_args = []
    kernel_arg_types = []
    for i, arg in enumerate(args):
        if kernel_params[i].is_constexpr:
            constants[kernel_params[i].name] = get_const_int(arg)
            continue
        if arg.dtype == "handle":
            assert isinstance(arg, tir.Var)
            elem_type = arg.type_annotation.element_type.dtype
            pointer_type = "*" + type_canonicalisation_dict[elem_type]
            signature[kernel_params[i].name] = pointer_type
        else:
            signature[kernel_params[i].name] = type_canonicalisation_dict[arg.dtype]
        kernel_args.append(arg)
        kernel_arg_types.append(arg.dtype)

    # TODO: Support default argument in the kernel
    # TODO: Add specialization for aligned buffer pointers
    source = triton.compiler.ASTSource(fn=func, constants=constants, signature=signature)
    compiled = triton.compiler.compile(source)
    return compiled, kernel_args, kernel_arg_types


def _pack_kernel_module(
    compiled: "triton.compiler.CompiledKernel",
    grid: List[Union[int, tir.PrimExpr]],
    kernel_arg_types: List[str],
) -> Tuple[Module, List[Union[int, tir.PrimExpr]]]:
    """Pack the compiled kernel into a TVM CUDAModule"""
    kernel_metadata = compiled.metadata
    kernel_name = compiled.name
    ptx = compiled.asm["ptx"]
    assert kernel_metadata.num_ctas == 1, "Cluster is not supported"
    num_warps = kernel_metadata.num_warps
    launch_param_tags = ["threadIdx.x"] + ["blockIdx.x", "blockIdx.y", "blockIdx.z"][: len(grid)]
    launch_args = [num_warps * 32] + list(grid)
    if kernel_metadata.shared > 0:
        # Add shared memory size to the launch arguments
        launch_param_tags.append("tir.use_dyn_shared_memory")
        launch_args.append(kernel_metadata.shared)
    tvm_metadata = """{{
        "tvm_version": "{version}",
        "func_info": {{
            "{kernel_name}": {{
                "name": "",
                "arg_types": {arg_types},
                "launch_param_tags": {launch_param_tags}
            }}
        }}
    }}""".format_map(
        {
            "version": __version__,
            "kernel_name": kernel_name,
            "arg_types": json.dumps(kernel_arg_types),
            "launch_param_tags": json.dumps(launch_param_tags),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ptx_path = f"{temp_dir}/{kernel_name}.ptx"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        with open(f"{temp_dir}/{kernel_name}.tvm_meta.json", "w") as f:
            f.write(tvm_metadata)
        kernel_module = load_module(ptx_path)

    return kernel_module, launch_args


def call_triton(
    func: "triton.runtime.jit.JITFunction",
    grid: List[Union[int, tir.PrimExpr]],
    *args: List[tir.PrimExpr],
    **kwargs: Any,
):
    """Invoke a triton kernel.

    This is a macro that bridges the Triton kernel with TVM runtime. Specifically, it performs the
    following steps:
        - Deduce the kernel signature and generate the Triton kernel
        - Embed the compiled kernel into the current IRModule as an external module
        - Generate a call to the Triton kernel following its calling convention via call_packed.

    Parameters
    ----------
    func : tt
        The Triton kernel function to invoke.
    grid : List[Union[int, tir.PrimExpr]]
        The grid configuration to launch the kernel.
    args : List[tir.PrimExpr]
        The arguments to the Triton kernel.
    kwargs : Any
        Additional options for the kernel compilation.
    """
    compiled, kernel_args, kernel_arg_types = _generate_triton_kernel(func, grid, *args, **kwargs)
    kernel_module, launch_args = _pack_kernel_module(compiled, grid, kernel_arg_types)
    from .ir import call_packed
    from ..ir import module_get_attr, module_set_attr

    kernel_launch = call_packed(compiled.name, *kernel_args, *launch_args)
    # Attach the kernel module to the current IRModule
    external_mods = module_get_attr("external_mods") or []
    external_mods.append(kernel_module)
    module_set_attr("external_mods", external_mods, True)
    return kernel_launch
