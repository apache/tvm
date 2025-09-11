# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""BasePyModule: Base class for IRModules with Python function support."""

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax, tir
from tvm.ir import IRModule
from tvm.runtime import Device, Tensor, PackedFunc
from tvm.target import Target

try:
    from torch.utils.dlpack import to_dlpack as to_dlpack_legacy
except ImportError:
    to_dlpack_legacy = None


class BasePyModule:
    """Base class that allows Python functions in IRModule with DLPack conversion.

    This class provides the infrastructure for:
    1. JIT compilation of TIR and Relax functions.
    2. DLPack-based conversion between PyTorch tensors and TVM Tensors.
    3. Wrapping Relax functions for easy Python calling.
    4. Cross-function calls between Python, TIR, and Relax functions.

    Only IRModules that inherit from this class are allowed to contain Python functions.
    """

    def __init__(
        self,
        ir_mod: IRModule,
        device: Device,
        target: Optional[Target] = None,
    ):
        """Initialize BasePyModule with JIT compilation and DLPack conversion."""
        self.device = device
        self.ir_mod = ir_mod

        # Delegate IRModule operations
        self.functions = ir_mod.functions
        self.attrs = ir_mod.attrs
        self.global_infos = ir_mod.global_infos
        self.__getitem__ = ir_mod.__getitem__
        self.__setitem__ = ir_mod.__setitem__
        self.functions_items = ir_mod.functions_items
        self.with_attr = ir_mod.with_attr
        self.get_attr = ir_mod.get_attr
        self.update_global_info = ir_mod.update_global_info

        def _getattr_python_function(name: str) -> Any:
            """Support direct attribute access to funcs and IRModule methods."""
            if name in self.pyfuncs:
                return self.pyfuncs[name]
            if name in self.compiled_tir_funcs:
                return self.compiled_tir_funcs[name]
            if self.relax_vm and name in self.relax_func_names:
                try:
                    return self.relax_vm[name]
                except AttributeError:  # More specific exception
                    return None
            if hasattr(self.ir_mod, name):
                return getattr(self.ir_mod, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        self.__getattr__ = _getattr_python_function

        self.compiled_tir_funcs: Dict[str, PackedFunc] = {}
        self.extern_funcs: Dict[str, PackedFunc] = {}
        self.tir_func_names: List[str] = []
        self.relax_func_names: List[str] = []
        self.relax_vm: Optional[relax.VirtualMachine] = None
        self.pyfuncs: Dict[str, Any] = {}

        if target is None:
            target = Target.from_device(device)
        elif isinstance(target, str):
            target = Target(target)
        self.target = target

        self._collect_function_names()
        self._compile_functions()
        self._wrap_tir_functions()
        self._wrap_relax_functions()

    def _collect_function_names(self):
        """Collect names of TIR and Relax functions from IRModule."""
        for global_var, func in self.ir_mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                self.tir_func_names.append(global_var.name_hint)
            elif isinstance(func, relax.Function):
                self.relax_func_names.append(global_var.name_hint)

    def _compile_functions(self):
        """Compile TIR and Relax functions using JIT compilation."""
        # Compile TIR functions first
        tir_mod = tvm.IRModule(
            {
                gv: func
                for gv, func in self.ir_mod.functions_items()
                if isinstance(func, tir.PrimFunc)
            }
        )
        if tir_mod:
            try:
                tir_exec_mod = tvm.compile(tir_mod, target=self.target)
                for func_name in self.tir_func_names:
                    self.compiled_tir_funcs[func_name] = tir_exec_mod[func_name]
            # pylint: disable=broad-exception-caught
            except Exception as error:
                print(f"Warning: Failed to compile one or more TIR functions: {error}")

        relax_mod = tvm.IRModule(
            {
                gv: func
                for gv, func in self.ir_mod.functions_items()
                if isinstance(func, relax.Function)
            }
        )
        if relax_mod:
            try:
                exec_mod = tvm.compile(self.ir_mod, target=self.target)
                self.relax_vm = relax.VirtualMachine(exec_mod, self.device)
            # pylint: disable=broad-exception-caught
            except Exception as error:
                print(f"Warning: Failed to compile Relax VM: {error}")
                self.relax_vm = None

    def _wrap_tir_functions(self):
        """Wrap TIR functions to make them accessible as instance attributes."""
        for func_name, func in self.compiled_tir_funcs.items():
            setattr(self, func_name, func)

    def _wrap_relax_functions(self):
        """Wrap Relax functions to be callable from Python with auto conversion."""
        for func_name in self.relax_func_names:

            def _create_relax_wrapper(name):
                def wrapper(*args, **kwargs):
                    """Wrapper for Relax function with automatic tensor conversion."""
                    if hasattr(self.ir_mod, "pyfuncs") and name in self.ir_mod.pyfuncs:
                        return self.ir_mod.pyfuncs[name](*args, **kwargs)

                    if self.relax_vm is not None:
                        converted_args = self._convert_pytorch_to_tvm(list(args))
                        converted_kwargs = {
                            k: self._convert_pytorch_to_tvm(v) for k, v in kwargs.items()
                        }
                        result = self.relax_vm[name](*converted_args, **converted_kwargs)
                        return self._convert_tvm_to_pytorch(result)

                    raise RuntimeError(
                        f"Neither converted Python function nor Relax VM available for {name}"
                    )

                wrapper.__name__ = name
                wrapper.__doc__ = f"Wrapped Relax function: {name}"
                return wrapper

            setattr(self, func_name, _create_relax_wrapper(func_name))

    def call_tir(self, tir_func, args, out_sinfo):
        """Call a TIR function with PyTorch tensors."""
        # Try to get function name from different sources
        if isinstance(tir_func, str):
            func_name = tir_func
        elif hasattr(tir_func, "name"):
            func_name = tir_func.name
        elif hasattr(tir_func, "__name__"):
            func_name = tir_func.__name__
        else:
            # Try to find by function object reference
            for name, func in self.compiled_tir_funcs.items():
                if func == tir_func:
                    func_name = name
                    break
            else:
                func_name = None

        if not func_name or func_name not in self.compiled_tir_funcs:
            available_funcs = list(self.compiled_tir_funcs.keys())
            raise ValueError(
                f"Could not resolve or find compiled TIR function: {tir_func}. "
                f"Available functions: {available_funcs}"
            )
        func = self.compiled_tir_funcs[func_name]

        out = self._create_output_tensors(out_sinfo, args)
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)

        func(*tvm_args, *tvm_out)

        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def call_dps_packed(self, func_name: str, args, out_sinfo):
        """Call a packed function with PyTorch tensors, converting TVM Tensors via DLPack."""
        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            return getattr(self, func_name)(*args)

        if func_name not in self.extern_funcs:
            try:
                self.extern_funcs[func_name] = tvm.get_global_func(func_name)
            except ValueError as error:
                raise ValueError(
                    f"Function '{func_name}' not found as a global function. "
                    f"Please implement it as a method or register it."
                ) from error
        func = self.extern_funcs[func_name]

        out = self._create_output_tensors(out_sinfo, args)
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)
        func(*tvm_args, *tvm_out)
        return out[0] if len(out) == 1 else out

    def call_py_func(self, func_name: str, args):
        """Call a Python function stored in the IRModule's pyfuncs."""
        if func_name not in self.ir_mod.pyfuncs:
            raise ValueError(f"Python function '{func_name}' not found in IRModule pyfuncs")
        py_func = self.ir_mod.pyfuncs[func_name]
        converted_args = self._convert_tvm_to_pytorch(args)
        return py_func(*converted_args)

    def _create_output_tensors(self, out_sinfo, in_args=None):
        # pylint: disable=import-outside-toplevel
        import torch

        sinfo_list = out_sinfo if isinstance(out_sinfo, list) else [out_sinfo]
        out_tensors = []
        for sinfo in sinfo_list:
            if isinstance(sinfo, (tuple, list)) and all(
                isinstance(x, (int, np.integer)) for x in sinfo
            ):
                out_tensors.append(torch.zeros(list(map(int, sinfo)), dtype=torch.float32))
                continue

            if hasattr(sinfo, "shape") and hasattr(sinfo, "dtype"):
                concrete_shape = self._infer_concrete_shape_from_args(sinfo.shape, in_args)
                torch_dtype = self._convert_tvm_dtype_to_torch(sinfo.dtype)
                out_tensors.append(torch.zeros(concrete_shape, dtype=torch_dtype))
                continue

            out_tensors.append(torch.zeros((1,), dtype=torch.float32))
        return out_tensors

    def _infer_concrete_shape_from_args(self, shape, in_args):

        concrete = []
        symbolic_positions = []
        for idx, dim in enumerate(shape):
            if isinstance(dim, (int, np.integer)):
                concrete.append(int(dim))
            elif isinstance(dim, tir.IntImm):
                concrete.append(int(dim.value))
            else:
                concrete.append(None)
                symbolic_positions.append(idx)

        if not symbolic_positions:
            return concrete

        candidates = []
        if in_args is not None:
            if not isinstance(in_args, (list, tuple)):
                in_args = [in_args]
            for obj in in_args:
                if hasattr(obj, "shape") and isinstance(obj.shape, (tuple, list)):
                    try:
                        candidates.append(tuple(int(x) for x in obj.shape))
                        continue
                    except (ValueError, TypeError):
                        # Skip objects with invalid shapes
                        pass

        target_ndim = len(shape)
        for cand in candidates:
            if len(cand) == target_ndim:
                for pos in symbolic_positions:
                    concrete[pos] = cand[pos]
                if all(x is not None for x in concrete):
                    return concrete

        raise ValueError(
            "Cannot infer concrete output shape from symbolic shape and inputs. "
            "Please provide a concrete `out_sinfo` (e.g., a tuple/list of ints) "
            "or ensure input tensors carry shapes that determine output extents."
        )

    def _convert_tvm_dtype_to_torch(self, tvm_dtype: str) -> "torch.dtype":
        """Convert TVM dtype string to PyTorch dtype."""
        # pylint: disable=import-outside-toplevel
        import torch

        dtype_mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        return dtype_mapping.get(str(tvm_dtype), torch.float32)

    def _convert_pytorch_to_tvm(
        self, tensors: Union[Any, List[Any], Tuple[Any, ...]]
    ) -> Union[Tensor, List[Tensor]]:
        """Convert PyTorch tensors to TVM Tensors using DLPack."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tensors, (list, tuple)):
            return [self._convert_single_pytorch_to_tvm(t) for t in tensors]
        return self._convert_single_pytorch_to_tvm(tensors)

    def _convert_single_pytorch_to_tvm(self, tensor: Any) -> Tensor:
        """Convert a single PyTorch tensor to TVM Tensor with robust fallbacks."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tensor, Tensor):
            return tensor
        if isinstance(tensor, torch.Tensor):
            # 1. Try modern `torch.to_dlpack` (preferred for PyTorch >= 1.7)
            try:
                dlpack = torch.to_dlpack(tensor)
                return tvm.runtime.from_dlpack(dlpack)
            except (AttributeError, ValueError):
                pass  # Fall through to the next method
            # 2. Try legacy `torch.utils.dlpack.to_dlpack`
            if to_dlpack_legacy:
                try:
                    dlpack = to_dlpack_legacy(tensor)
                    return tvm.runtime.from_dlpack(dlpack)
                except (AttributeError, ValueError) as error_legacy:
                    print(
                        f"Warning: Legacy DLPack conversion failed ({error_legacy}), "
                        f"using numpy fallback."
                    )
            # 3. If all DLPack methods fail, use numpy fallback
            numpy_array = tensor.detach().cpu().numpy()
            return tvm.runtime.tensor(numpy_array, device=self.device)

        # For other types (like scalars, lists), convert to numpy first
        try:
            numpy_array = np.array(tensor, dtype=np.float32)
            return tvm.runtime.tensor(numpy_array, device=self.device)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Unsupported type for conversion to TVM Tensor: {type(tensor)}"
            ) from error

    def _convert_tvm_to_pytorch(
        self, tvm_arrays: Union[Any, List[Any]]
    ) -> Union["torch.Tensor", List["torch.Tensor"]]:
        """Convert TVM Tensors to PyTorch tensors using DLPack."""
        if isinstance(tvm_arrays, (list, tuple)):
            return [self._convert_single_tvm_to_pytorch(arr) for arr in tvm_arrays]
        return self._convert_single_tvm_to_pytorch(tvm_arrays)

    def _convert_single_tvm_to_pytorch(self, tvm_array: Any) -> "torch.Tensor":
        """Convert a single TVM Tensor to PyTorch tensor using DLPack."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tvm_array, torch.Tensor):
            return tvm_array
        if not isinstance(tvm_array, Tensor):
            return torch.tensor(tvm_array)
        try:
            return torch.from_dlpack(tvm_array)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            print(f"Warning: DLPack conversion from TVM failed ({error}), using numpy fallback")
            numpy_array = tvm_array.numpy()
            return torch.from_numpy(numpy_array)

    def get_function(self, name: str) -> Optional[PackedFunc]:
        """Get a compiled function by name."""
        if name in self.compiled_tir_funcs:
            return self.compiled_tir_funcs[name]
        if name in self.extern_funcs:
            return self.extern_funcs[name]
        if self.relax_vm and name in self.relax_func_names:
            try:
                if hasattr(self, name):
                    return getattr(self, name)
                return self.relax_vm[name]
            except AttributeError as error:
                print(f"Warning: Failed to get Relax function '{name}': {error}")
        return None

    def list_functions(self) -> Dict[str, List[str]]:
        """List all available functions."""
        return {
            "tir": self.tir_func_names,
            "relax": self.relax_func_names,
            "extern": list(self.extern_funcs.keys()),
        }

    def add_python_function(self, name: str, func: callable):
        """Add a Python function to the module."""
        self.pyfuncs[name] = func

        # Create a wrapper that handles both instance methods and static functions
        # pylint: disable=import-outside-toplevel
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if params and params[0] == "self":
                return func(self, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Set the wrapper as an instance attribute
        setattr(self, name, wrapper)

    def script(
        self,
        *,
        name: Optional[str] = None,
        show_meta: bool = False,
        ir_prefix: str = "I",
        tir_prefix: str = "T",
        relax_prefix: str = "R",
        module_alias: str = "cls",
        buffer_dtype: str = "float32",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        syntax_sugar: bool = True,
        show_object_address: bool = False,
        show_all_struct_info: bool = True,
    ) -> str:
        """Print TVM IR into TVMScript text format with Python function support.

        This method extends the standard IRModule script() method to handle
        Python functions stored in the IRModule's pyfuncs attribute.
        """
        # First get the standard IRModule script
        base_script = self.ir_mod.script(
            name=name,
            show_meta=show_meta,
            ir_prefix=ir_prefix,
            tir_prefix=tir_prefix,
            relax_prefix=relax_prefix,
            module_alias=module_alias,
            buffer_dtype=buffer_dtype,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            verbose_expr=verbose_expr,
            indent_spaces=indent_spaces,
            print_line_numbers=print_line_numbers,
            num_context_lines=num_context_lines,
            syntax_sugar=syntax_sugar,
            show_object_address=show_object_address,
            show_all_struct_info=show_all_struct_info,
        )

        # If there are no Python functions, return the base script
        if not hasattr(self.ir_mod, "pyfuncs") or not self.ir_mod.pyfuncs:
            return base_script

        # Insert Python functions into the script
        return self._insert_python_functions(base_script, indent_spaces)

    def _insert_python_functions(self, base_script: str, indent_spaces: int) -> str:
        """Insert Python functions into the TVMScript output."""
        lines = base_script.split("\n")
        result_lines = []

        # Find the class definition line and insert Python functions after it
        class_found = False
        class_indent = 0

        for line in lines:
            result_lines.append(line)

            # Look for class definition
            if not class_found and line.strip().startswith("class "):
                class_found = True
                class_indent = len(line) - len(line.lstrip())

                # Insert Python functions after the class definition
                if hasattr(self.ir_mod, "pyfuncs") and self.ir_mod.pyfuncs:
                    for func_name, func in self.ir_mod.pyfuncs.items():
                        # Get the function source code
                        func_source = self._get_function_source(func)
                        if func_source:
                            # Format the function with proper indentation
                            formatted_func = self._format_python_function(
                                func_name, func_source, class_indent + indent_spaces
                            )
                            result_lines.append(formatted_func)
                            result_lines.append("")  # Add empty line for separation

        return "\n".join(result_lines)

    def _get_function_source(self, func: callable) -> Optional[str]:
        """Get the source code of a Python function."""
        try:
            source = inspect.getsource(func)
            return source
        except (OSError, TypeError):
            # If we can't get the source, return None
            return None

    def _format_python_function(self, _func_name: str, func_source: str, indent: int) -> str:
        """Format a Python function with proper indentation for TVMScript."""
        lines = func_source.split("\n")
        formatted_lines = []

        for line in lines:
            # Skip the function definition line if it's already properly indented
            if line.strip().startswith("def ") or line.strip().startswith("@"):
                # Keep decorators and function definition as is
                formatted_lines.append(" " * indent + line.strip())
            else:
                # Add proper indentation for the function body
                formatted_lines.append(" " * indent + line.strip())

        return "\n".join(formatted_lines)

    def show(
        self, style: Optional[str] = None, black_format: Optional[bool] = None, **kwargs
    ) -> None:
        """A sugar for print highlighted TVM script with Python function support.

        This method extends the standard IRModule show() method to handle
        Python functions stored in the IRModule's pyfuncs attribute.
        """
        from tvm.script.highlight import cprint  # pylint: disable=import-outside-toplevel

        if black_format is None:
            env = os.environ.get("TVM_BLACK_FORMAT")
            black_format = env and int(env)

        script_content = self.script(**kwargs)
        cprint(script_content, style=style, black_format=black_format)
