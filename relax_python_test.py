from typing import Optional

import torch
import torch.nn.functional as F

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


class BasePyModule:
    def __init__(
        self,
        ir_mod: tvm.IRModule,
        device: tvm.runtime.Device,
        target: Optional[tvm.target.Target] = None,
    ):
        self.compiled_tir_funcs = {}
        self.extern_funcs = {}
        self.tir_func_names = []
        self.relax_func_names = []
        self.relax_vm = None

        # Compile all the TIR functions in the class.
        if target is None:
            target = tvm.target.Target.from_device(device)

        # Apply pass that updates all TIR functions to be public, with global symbols attached.
        # ir_mod = VisibilityUpdater()(ir_mod)

        for gv, func in ir_mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                self.tir_func_names.append(gv.name_hint)
            elif isinstance(func, relax.Function):
                self.relax_func_names.append(gv.name_hint)

        # Compile the IRModule Relax and TIR functions in the IRModule.
        # TIR scheduling will be done with dlight rules in the relax pipeline.
        exec = tvm.compile(
            ir_mod,
            target=target,
            relax_pipeline=relax.get_default_pipeline(target),
            tir_pipeline=tir.get_default_tir_pipeline(target),
        )
        self.relax_vm = relax.VirtualMachine(exec, device)

        # Register the wrapped function to the class,
        # so that it can be called like a normal python function
        # with torch tensor arguments and return values.
        for func_name in self.relax_func_names:

            def _wrap_relax_func(*args):
                # Convert args to tvm ndarray with dlpack...
                # args = ...
                out = self.relax_vm[func_name](*args)
                # Convert out to torch tensor...
                # out = ...
                return out

            setattr(self, func_name, _wrap_relax_func)

        # Lookup compiled TIR functions from the VM
        for func_name in self.tir_func_names:
            self.compiled_tir_funcs[func_name] = self.relax_vm[func_name]

    def call_tir(self, tir_func, args, out_sinfo):
        """Call a TIR function with PyTorch tensors, converting to/from TVM NDArrays via DLPack."""
        # Create output tensors based on out_sinfo
        out = (
            [torch.empty(out_sinfo.shape, dtype=out_sinfo.dtype)]
            if not isinstance(out_sinfo, list)
            else [torch.empty(sinfo.shape, dtype=sinfo.dtype) for sinfo in out_sinfo]
        )

        if not isinstance(tir_func, tir.PrimFunc):
            raise ValueError(f"Input function {tir_func} is not a tir.PrimFunc")
        func = self.compiled_tir_funcs[tir_func.__name__]

        # Convert PyTorch tensors to TVM NDArrays via DLPack
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)

        # Call the TIR function
        func(*tvm_args, *tvm_out)
        
        # Convert output back to PyTorch tensors
        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def call_dps_packed(self, func_name, args, out_sinfo):
        """Call a packed function with PyTorch tensors, converting to/from TVM NDArrays via DLPack."""
        # Create output tensors based on out_sinfo
        out = (
            [torch.empty(out_sinfo.shape, dtype=out_sinfo.dtype)]
            if not isinstance(out_sinfo, list)
            else [torch.empty(sinfo.shape, dtype=sinfo.dtype) for sinfo in out_sinfo]
        )

        if func_name not in self.extern_funcs:
            func = tvm.get_global_func(func_name)
            self.extern_funcs[func_name] = func
        else:
            func = self.extern_funcs[func_name]

        # Convert PyTorch tensors to TVM NDArrays via DLPack
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)

        # Call the packed function
        func(*tvm_args, *tvm_out)
        
        # Convert output back to PyTorch tensors
        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def _convert_pytorch_to_tvm(self, tensors):
        """Convert PyTorch tensors to TVM NDArrays using DLPack.
        
        Parameters
        ----------
        tensors : Union[torch.Tensor, List[torch.Tensor]]
            PyTorch tensor(s) to convert.
            
        Returns
        -------
        Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]
            TVM NDArray(s) converted from PyTorch tensors.
        """
        if isinstance(tensors, list):
            return [self._convert_single_pytorch_to_tvm(t) for t in tensors]
        else:
            return self._convert_single_pytorch_to_tvm(tensors)
    
    def _convert_single_pytorch_to_tvm(self, tensor):
        """Convert a single PyTorch tensor to TVM NDArray using DLPack.
        
        Parameters
        ----------
        tensor : torch.Tensor
            PyTorch tensor to convert.
            
        Returns
        -------
        tvm.nd.NDArray
            TVM NDArray converted from PyTorch tensor.
        """
        try:
            # Use DLPack for efficient conversion
            if hasattr(tensor, 'to_dlpack'):
                # PyTorch 1.10+ supports to_dlpack
                dlpack = tensor.to_dlpack()
                tvm_tensor = tvm.nd.from_dlpack(dlpack)
                return tvm_tensor
            else:
                # Fallback: convert to numpy then to TVM
                numpy_array = tensor.detach().cpu().numpy()
                tvm_tensor = tvm.nd.array(numpy_array, device=self.device)
                return tvm_tensor
        except Exception as e:
            print(f"Warning: DLPack conversion failed, using fallback method: {e}")
            # Fallback: convert to numpy then to TVM
            numpy_array = tensor.detach().cpu().numpy()
            tvm_tensor = tvm.nd.array(numpy_array, device=self.device)
            return tvm_tensor
    
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
        """Convert a single TVM NDArray to PyTorch tensor using DLPack.
        
        Parameters
        ----------
        tvm_array : tvm.nd.NDArray
            TVM NDArray to convert.
            
        Returns
        -------
        torch.Tensor
            PyTorch tensor converted from TVM NDArray.
        """
        try:
            # Use DLPack for efficient conversion
            dlpack = tvm_array.to_dlpack()
            torch_tensor = torch.from_dlpack(dlpack)
            return torch_tensor
        except Exception as e:
            print(f"Warning: DLPack conversion failed, using fallback method: {e}")
            # Fallback: convert to numpy then to PyTorch
            numpy_array = tvm_array.numpy()
            torch_tensor = torch.from_numpy(numpy_array)
            return torch_tensor


@I.ir_module
class IRModuleWithPyFunc(BasePyModule):
    """Example IRModule with Python function.
    The base class BasePyModule implements the logic of cross-function calls
    and JIT compilation in Python.
    We only allow Python functions in IRModules that subclass the BasePyModule.
    """

    @I.pyfunc
    def main(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        lv = self.call_tir(self.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
        lv1 = F.relu(lv)
        lv2 = self.call_dps_packed("my_softmax", [lv1, 1], out_sinfo=R.Tensor((n, 20), "float32"))
        lv3 = self.my_identity_func(lv2)
        gv = lv3
        return gv

    @T.prim_func
    def matmul(
        var_A: T.handle,
        var_B: T.handle,
        var_C: T.handle,
    ):
        n = T.int32()
        A = T.match_buffer(var_A, (n, 16), "float32")
        B = T.match_buffer(var_B, (16, 20), "float32")
        C = T.match_buffer(var_C, (n, 20), "float32")
        for i, j, k in T.grid(n, 20, 16):
            with T.block("block"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def my_identity_func(x: R.Tensor(("n", 20), "float32")) -> R.Tensor(("n", 20), "float32"):
        return x

    # @R.function
    # def my_relax_func(
    #     x: R.Tensor(("n", 16), "float32"), w: R.Tensor((16, 20), "float32")
    # ) -> R.Tensor(("n", 20), "float32"):
    #     cls = IRModuleWithPyFunc
    #     n = T.int64()
    #     with R.dataflow():
    #         lv = R.call_py_func(cls.main)
    #     return x


def main():
    mod = IRModuleWithPyFunc
    print(mod.script())


if __name__ == "__main__":
    main()
