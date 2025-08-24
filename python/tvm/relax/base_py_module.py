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

from typing import Any, Dict, List, Optional, Union

import tvm
from tvm import relax, tir
from tvm.ir import IRModule
from tvm.runtime import Device, PackedFunc
from tvm.target import Target


class BasePyModule:
    """Base class that allows Python functions in IRModule with DLPack conversion.
    
    This class provides the infrastructure for:
    1. JIT compilation of TIR and Relax functions
    2. DLPack-based conversion between PyTorch tensors and TVM NDArrays
    3. Wrapping Relax functions for easy Python calling
    4. Cross-function calls between Python, TIR, and Relax functions
    
    Only IRModules that inherit from this class are allowed to contain Python functions.
    """

    def __init__(
        self,
        ir_mod: IRModule,
        device: Device,
        target: Optional[Target] = None,
    ):
        """Initialize BasePyModule with JIT compilation and DLPack conversion.
        
        Parameters
        ----------
        ir_mod : IRModule
            The IRModule containing TIR and Relax functions to compile.
        device : Device
            The target device for execution.
        target : Optional[Target]
            The compilation target. If None, inferred from device.
        """
        self.device = device
        self.ir_mod = ir_mod
        self.compiled_tir_funcs: Dict[str, PackedFunc] = {}
        self.extern_funcs: Dict[str, PackedFunc] = {}
        self.tir_func_names: List[str] = []
        self.relax_func_names: List[str] = []
        self.relax_vm: Optional[relax.VirtualMachine] = None

        # Set target if not provided
        if target is None:
            target = Target.from_device(device)
        self.target = target

        # Collect function names from IRModule
        self._collect_function_names()
        
        # Perform JIT compilation
        self._compile_functions()
        
        # Wrap Relax functions for easy calling
        self._wrap_relax_functions()

    def _collect_function_names(self):
        """Collect names of TIR and Relax functions from IRModule."""
        for gv, func in self.ir_mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                self.tir_func_names.append(gv.name_hint)
            elif isinstance(func, relax.Function):
                self.relax_func_names.append(gv.name_hint)
        
        print(f"✓ Collected {len(self.tir_func_names)} TIR functions: {self.tir_func_names}")
        print(f"✓ Collected {len(self.relax_func_names)} Relax functions: {self.relax_func_names}")

    def _compile_functions(self):
        """Compile TIR and Relax functions using JIT compilation."""
        print(f"🔨 Compiling IRModule for target: {self.target}")
        
        try:
            # First, try to compile TIR functions separately for better access
            print(f"  Attempting separate TIR compilation...")
            
            # Extract TIR functions from IRModule
            tir_mod = tvm.IRModule()
            for gv, func in self.ir_mod.functions_items():
                if isinstance(func, tir.PrimFunc):
                    tir_mod[gv] = func
            
            if len(tir_mod.functions) > 0:
                try:
                    # Compile TIR functions separately
                    tir_exec_mod = tvm.build(tir_mod, target=self.target)
                    print(f"  TIR compilation successful: {type(tir_exec_mod)}")
                    
                    # Store compiled TIR functions
                    for func_name in self.tir_func_names:
                        try:
                            func = tir_exec_mod[func_name]
                            self.compiled_tir_funcs[func_name] = func
                            print(f"  ✓ TIR function '{func_name}' compiled successfully")
                        except Exception as e:
                            print(f"  ⚠ Warning: Failed to get TIR function '{func_name}': {e}")
                except Exception as e:
                    print(f"  ⚠ Warning: Separate TIR compilation failed: {e}")
            
            # Now compile the full IRModule for Relax functions
            print(f"  Compiling full IRModule for Relax functions...")
            exec_mod = tvm.compile(
                self.ir_mod,
                target=self.target,
                relax_pipeline=relax.get_default_pipeline(self.target),
                tir_pipeline=tir.get_default_tir_pipeline(self.target),
            )
            
            print(f"  Full compilation successful: {type(exec_mod)}")
            
            # Create Relax Virtual Machine for Relax functions
            self.relax_vm = relax.VirtualMachine(exec_mod, self.device)
            
            print("✓ JIT compilation completed")
            
        except Exception as e:
            print(f"✗ Error during compilation: {e}")
            import traceback
            traceback.print_exc()
            self.relax_vm = None
            print("✓ JIT compilation failed, but continuing...")

    def _wrap_relax_functions(self):
        """Wrap Relax functions to make them callable from Python with automatic conversion."""
        if self.relax_vm is None:
            print(f"  ⚠ Warning: Relax VM not available, skipping function wrapping")
            return
            
        for func_name in self.relax_func_names:
            # Create a wrapper that handles tensor conversion
            def _create_relax_wrapper(name):
                def wrapper(*args, **kwargs):
                    """Wrapper for Relax function with automatic tensor conversion."""
                    try:
                        # Convert PyTorch tensors to TVM NDArrays if needed
                        converted_args = self._convert_pytorch_to_tvm(args)
                        converted_kwargs = {k: self._convert_pytorch_to_tvm(v) for k, v in kwargs.items()}
                        
                        # Call the Relax function
                        result = self.relax_vm[name](*converted_args, **converted_kwargs)
                        
                        # Convert result back to PyTorch tensors if needed
                        return self._convert_tvm_to_pytorch(result)
                    except Exception as e:
                        print(f"Error calling Relax function '{name}': {e}")
                        raise
                
                wrapper.__name__ = name
                wrapper.__doc__ = f"Wrapped Relax function: {name}"
                return wrapper
            
            # Set the wrapped function as an attribute
            setattr(self, func_name, _create_relax_wrapper(func_name))
            print(f"  ✓ Relax function '{func_name}' wrapped for Python calling")

    def call_tir(self, tir_func, args, out_sinfo):
        """Call a TIR function with PyTorch tensors, converting to/from TVM NDArrays via DLPack.
        
        Parameters
        ----------
        tir_func : Union[tir.PrimFunc, str, PackedFunc]
            The TIR function to call. Can be a function object, function name, or compiled function.
        args : Union[torch.Tensor, List[torch.Tensor]]
            Input PyTorch tensors.
        out_sinfo : Union[R.Tensor, List[R.Tensor]]
            Output shape and type information.
            
        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Output PyTorch tensors.
        """
        # Get the compiled function - handle different input types
        if isinstance(tir_func, str):
            # Function name provided
            func_name = tir_func
            if func_name not in self.compiled_tir_funcs:
                raise ValueError(f"TIR function '{func_name}' not found in compiled functions")
            func = self.compiled_tir_funcs[func_name]
        elif hasattr(tir_func, 'name') and tir_func.name in self.compiled_tir_funcs:
            # TIR function object with name
            func_name = tir_func.name
            func = self.compiled_tir_funcs[func_name]
        elif tir_func in self.compiled_tir_funcs.values():
            # Already a compiled function
            func = tir_func
        else:
            # Try to find by function name
            func_name = getattr(tir_func, 'name', None) or getattr(tir_func, '__name__', None)
            if func_name and func_name in self.compiled_tir_funcs:
                func = self.compiled_tir_funcs[func_name]
            else:
                raise ValueError(f"Could not resolve TIR function: {tir_func}")
        
        # Create output tensors based on out_sinfo
        out = self._create_output_tensors(out_sinfo)
        
        # Convert PyTorch tensors to TVM NDArrays via DLPack
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)
        
        # Call the TIR function
        func(*tvm_args, *tvm_out)
        
        # Convert output back to PyTorch tensors
        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def call_dps_packed(self, func_name: str, args, out_sinfo):
        """Call a packed function with PyTorch tensors, converting to/from TVM NDArrays via DLPack.
        
        Parameters
        ----------
        func_name : str
            Name of the packed function to call.
        args : Union[torch.Tensor, List[torch.Tensor]]
            Input PyTorch tensors.
        out_sinfo : Union[R.Tensor, List[R.Tensor]]
            Output shape and type information.
            
        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Output PyTorch tensors.
        """
        # Get or create the packed function
        if func_name not in self.extern_funcs:
            try:
                func = tvm.get_global_func(func_name)
                self.extern_funcs[func_name] = func
            except Exception as e:
                raise ValueError(f"Failed to get global function '{func_name}': {e}")
        else:
            func = self.extern_funcs[func_name]
        
        # Create output tensors based on out_sinfo
        out = self._create_output_tensors(out_sinfo)
        
        # Convert PyTorch tensors to TVM NDArrays via DLPack
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)
        
        # Call the packed function
        func(*tvm_args, *tvm_out)
        
        # Convert output back to PyTorch tensors
        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def call_py_func(self, func_name: str, args):
        """Call a Python function stored in the IRModule's pyfuncs.
        
        This method provides true PyTorch input/output support:
        - Input: TVM NDArrays are converted to PyTorch tensors
        - Output: PyTorch tensors are returned directly (not converted back)
        
        Parameters
        ----------
        func_name : str
            The name of the Python function to call.
        args : List
            The arguments to pass to the Python function (TVM NDArrays).
            
        Returns
        -------
        torch.Tensor or List[torch.Tensor]
            The result of the Python function call as PyTorch tensor(s).
        """
        # Check if the function exists in pyfuncs
        if func_name not in self.ir_mod.pyfuncs:
            raise ValueError(f"Python function '{func_name}' not found in IRModule pyfuncs")
        
        # Get the Python function
        py_func = self.ir_mod.pyfuncs[func_name]
        
        # Convert TVM NDArrays to PyTorch tensors
        converted_args = self._convert_tvm_to_pytorch(args)
        
        # Call the Python function with PyTorch tensors
        result = py_func(*converted_args)
        
        # Return PyTorch tensor directly (don't convert back to TVM)
        # This ensures true PyTorch output as specified in the Motivation
        return result

    def _create_output_tensors(self, out_sinfo):
        """Create output PyTorch tensors based on shape and type information."""
        try:
            import torch
            
            if not isinstance(out_sinfo, list):
                out_sinfo = [out_sinfo]
            
            out_tensors = []
            for sinfo in out_sinfo:
                # Extract shape and dtype from R.Tensor
                if hasattr(sinfo, 'shape') and hasattr(sinfo, 'dtype'):
                    shape = sinfo.shape
                    dtype = sinfo.dtype
                    
                    # Convert TVM dtype to PyTorch dtype
                    torch_dtype = self._convert_tvm_dtype_to_torch(dtype)
                    
                    # Create empty tensor
                    out_tensor = torch.empty(shape, dtype=torch_dtype)
                    out_tensors.append(out_tensor)
                else:
                    # Fallback: create tensor with default dtype and shape
                    if hasattr(sinfo, 'shape'):
                        shape = sinfo.shape
                    else:
                        shape = (1,)  # Default shape
                    out_tensor = torch.empty(shape, dtype=torch.float32)
                    out_tensors.append(out_tensor)
            
            return out_tensors
            
        except ImportError:
            raise ImportError("PyTorch is required for output tensor creation")

    def _convert_tvm_dtype_to_torch(self, tvm_dtype):
        """Convert TVM dtype to PyTorch dtype."""
        try:
            import torch
            
            dtype_mapping = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
                "bool": torch.bool,
            }
            
            if isinstance(tvm_dtype, str):
                return dtype_mapping.get(tvm_dtype, torch.float32)
            elif hasattr(tvm_dtype, 'name'):
                return dtype_mapping.get(tvm_dtype.name, torch.float32)
            else:
                return torch.float32
                
        except ImportError:
            raise ImportError("PyTorch is required for dtype conversion")

    def _convert_pytorch_to_tvm(self, tensors):
        """Convert PyTorch tensors to TVM NDArrays using DLPack.
        
        Parameters
        ----------
        tensors : Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            PyTorch tensor(s) to convert.
            
        Returns
        -------
        Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]
            TVM NDArray(s) converted from PyTorch tensors.
        """
        if isinstance(tensors, (list, tuple)):
            return [self._convert_single_pytorch_to_tvm(t) for t in tensors]
        else:
            return self._convert_single_pytorch_to_tvm(tensors)
    
    def _convert_single_pytorch_to_tvm(self, tensor):
        """Convert a single PyTorch tensor to TVM NDArray using DLPack."""
        try:
            import torch
            
            # If it's already a TVM NDArray, return as is
            if hasattr(tensor, 'numpy') and hasattr(tensor, 'device'):
                return tensor
            
            # If it's a PyTorch tensor, convert using DLPack
            if isinstance(tensor, torch.Tensor):
                # Use DLPack for efficient conversion
                if hasattr(tensor, 'to_dlpack'):
                    try:
                        # PyTorch 1.10+ supports to_dlpack
                        dlpack = tensor.to_dlpack()
                        tvm_tensor = tvm.nd.from_dlpack(dlpack)
                        return tvm_tensor
                    except Exception as e:
                        print(f"Warning: DLPack conversion failed, using fallback method: {e}")
                
                # Fallback: convert to numpy then to TVM
                numpy_array = tensor.detach().cpu().numpy()
                tvm_tensor = tvm.nd.array(numpy_array, device=self.device)
                return tvm_tensor
            
            # Otherwise, try to convert to numpy first
            import numpy as np
            if hasattr(tensor, 'numpy'):
                numpy_array = tensor.numpy()
            else:
                # Ensure numpy array has a valid dtype
                numpy_array = np.array(tensor, dtype=np.float32)
            return tvm.nd.array(numpy_array, device=self.device)
            
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion")
    
    def _convert_tvm_to_pytorch(self, tvm_arrays):
        """Convert TVM NDArrays to PyTorch tensors using DLPack.
        
        Parameters
        ----------
        tvm_arrays : Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]
            TVM NDArray(s) to convert.
            
        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            PyTorch tensor(s) converted from TVM NDArrays.
        """
        if isinstance(tvm_arrays, list):
            return [self._convert_single_tvm_to_pytorch(arr) for arr in tvm_arrays]
        else:
            return self._convert_single_tvm_to_pytorch(tvm_arrays)
    
    def _convert_single_tvm_to_pytorch(self, tvm_array):
        """Convert a single TVM NDArray to PyTorch tensor using DLPack."""
        try:
            import torch
            
            # Use DLPack for efficient conversion
            try:
                dlpack = tvm_array.to_dlpack()
                torch_tensor = torch.from_dlpack(dlpack)
                return torch_tensor
            except Exception as e:
                print(f"Warning: DLPack conversion failed, using fallback method: {e}")
            
            # Fallback: convert to numpy then to PyTorch
            numpy_array = tvm_array.numpy()
            torch_tensor = torch.from_numpy(numpy_array)
            return torch_tensor
            
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion")

    def get_function(self, name: str) -> Optional[PackedFunc]:
        """Get a compiled function by name.
        
        Parameters
        ----------
        name : str
            Name of the function to retrieve.
            
        Returns
        -------
        Optional[PackedFunc]
            The compiled function, or None if not found.
        """
        if name in self.compiled_tir_funcs:
            return self.compiled_tir_funcs[name]
        elif name in self.extern_funcs:
            return self.extern_funcs[name]
        elif self.relax_vm and name in self.relax_func_names:
            # For Relax functions, return a wrapper that can be called
            try:
                # Return the wrapped function that's already set as an attribute
                if hasattr(self, name):
                    return getattr(self, name)
                else:
                    # If not wrapped, try to get from VM directly
                    return self.relax_vm[name]
            except Exception as e:
                print(f"Warning: Failed to get Relax function '{name}': {e}")
                return None
        else:
            return None

    def list_functions(self) -> Dict[str, List[str]]:
        """List all available functions.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping function types to function names.
        """
        return {
            "tir": self.tir_func_names,
            "relax": self.relax_func_names,
            "extern": list(self.extern_funcs.keys())
        }
