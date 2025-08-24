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
"""Python printer for Relax functions with PyTorch operator mapping."""

from typing import Dict, List, Optional, Union, Any
import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.relax import Function, Call, Var, Constant, Tuple, TupleGetItem
from tvm.relax import ShapeExpr, PrimValue, DataTypeImm, StringImm
from tvm.relax import If, BindingBlock, VarBinding, DataflowBlock
from tvm.relax import MatchCast, Binding
from tvm.relax.struct_info import TensorStructInfo, ShapeStructInfo, PrimStructInfo
from tvm.relax.struct_info import TupleStructInfo, ObjectStructInfo
from tvm.runtime.script_printer import PrinterConfig


class RelaxToPythonPrinter:
    """Convert Relax functions to executable Python code with PyTorch operator mapping."""
    
    def __init__(self):
        # Relax to PyTorch operator mapping
        self.op_mapping = {
            # Basic arithmetic operations
            "relax.add": "torch.add",
            "relax.subtract": "torch.sub",
            "relax.multiply": "torch.mul",
            "relax.divide": "torch.div",
            "relax.power": "torch.pow",
            "relax.floor_divide": "torch.floor_divide",
            "relax.mod": "torch.remainder",
            
            # Comparison operations
            "relax.equal": "torch.eq",
            "relax.greater": "torch.gt",
            "relax.greater_equal": "torch.ge",
            "relax.less": "torch.lt",
            "relax.less_equal": "torch.le",
            "relax.not_equal": "torch.ne",
            
            # Logical operations
            "relax.logical_and": "torch.logical_and",
            "relax.logical_or": "torch.logical_or",
            "relax.logical_not": "torch.logical_not",
            
            # Mathematical functions
            "relax.abs": "torch.abs",
            "relax.ceil": "torch.ceil",
            "relax.cos": "torch.cos",
            "relax.cosh": "torch.cosh",
            "relax.exp": "torch.exp",
            "relax.floor": "torch.floor",
            "relax.log": "torch.log",
            "relax.log2": "torch.log2",
            "relax.log10": "torch.log10",
            "relax.negative": "torch.neg",
            "relax.round": "torch.round",
            "relax.sin": "torch.sin",
            "relax.sinh": "torch.sinh",
            "relax.sqrt": "torch.sqrt",
            "relax.tan": "torch.tan",
            "relax.tanh": "torch.tanh",
            
            # Tensor operations
            "relax.reshape": "torch.reshape",
                    "relax.permute_dims": "torch.transpose",
            "relax.expand_dims": "torch.unsqueeze",
            "relax.squeeze": "torch.squeeze",
            "relax.concat": "torch.cat",
            "relax.split": "torch.split",
            "relax.take": "torch.index_select",
            "relax.strided_slice": "torch.narrow",
            
            # Reduction operations
            "relax.sum": "torch.sum",
            "relax.mean": "torch.mean",
            "relax.max": "torch.max",
            "relax.min": "torch.min",
            "relax.prod": "torch.prod",
            "relax.std": "torch.std",
            "relax.variance": "torch.var",
            
            # Neural network operations
            "relax.nn.conv2d": "torch.nn.functional.conv2d",
            "relax.nn.conv2d_transpose": "torch.nn.functional.conv_transpose2d",
            "relax.nn.avg_pool2d": "torch.nn.functional.avg_pool2d",
            "relax.nn.max_pool2d": "torch.nn.functional.max_pool2d",
            "relax.nn.adaptive_avg_pool2d": "torch.nn.functional.adaptive_avg_pool2d",
            "relax.nn.adaptive_max_pool2d": "torch.nn.functional.adaptive_max_pool2d",
            "relax.nn.softmax": "torch.nn.functional.softmax",
            "relax.nn.log_softmax": "torch.nn.functional.log_softmax",
            "relax.nn.relu": "torch.nn.functional.relu",
            "relax.nn.gelu": "torch.nn.functional.gelu",
            "relax.nn.sigmoid": "torch.nn.functional.sigmoid",
            "relax.nn.tanh": "torch.nn.functional.tanh",
            "relax.nn.dropout": "torch.nn.functional.dropout",
            "relax.nn.batch_norm": "torch.nn.functional.batch_norm",
            "relax.nn.layer_norm": "torch.nn.functional.layer_norm",
            "relax.nn.linear": "torch.nn.functional.linear",
            
                    # Special operations
        "relax.call_tir": "self._call_tir_wrapper",
        "relax.call_dps_packed": "self._call_dps_packed_wrapper",
        "relax.print": "print",
        "relax.call_py_func": "self._call_py_func_wrapper",
            
            # Shape inspection operations
            "relax.inspect.tensor_shape_i": "shape_access",
        }
        
        # Shape variable mapping for symbolic shapes
        self.shape_vars = {}
        
        # Generated Python code
        self.python_code = []
        self.indent_level = 0
        
    def print_relax_function(self, func: Function, func_name: str = None) -> str:
        """Convert a Relax function to Python code.
        
        Parameters
        ----------
        func : Function
            The Relax function to convert.
        func_name : str, optional
            Name for the generated Python function.
            
        Returns
        -------
        str
            Generated Python code.
        """
        if func_name is None:
            func_name = func.name_hint if hasattr(func, 'name_hint') else "relax_function"
        
        # Reset state
        self.python_code = []
        self.indent_level = 0
        self.shape_vars = {}
        
        # Generate function signature
        self._print_function_signature(func, func_name)
        
        # Generate function body
        self._print_function_body(func)
        
        # Join all lines
        return "\n".join(self.python_code)
    
    def _print_function_signature(self, func: Function, func_name: str):
        """Print function signature with proper type annotations."""
        # Function decorator
        self.python_code.append("@torch.jit.script")
        
        # Function definition
        params = []
        for param in func.params:
            param_name = param.name_hint
            param_type = self._get_python_type_annotation(param.struct_info)
            params.append(f"{param_name}: {param_type}")
        
        # Return type
        if hasattr(func, 'ret_struct_info') and func.ret_struct_info:
            ret_type = self._get_python_type_annotation(func.ret_struct_info)
            signature = f"def {func_name}({', '.join(params)}) -> {ret_type}:"
        else:
            signature = f"def {func_name}({', '.join(params)}):"
        
        self.python_code.append(signature)
    
    def _print_function_body(self, func: Function):
        """Print function body by visiting all bindings."""
        self.indent_level += 1
        
        # Visit all bindings in the function
        if func.body:
            if hasattr(func.body, 'blocks'):
                # This is a SeqExpr with blocks
                for block in func.body.blocks:
                    self._visit_binding_block(block)
                # Handle the final body expression
                if hasattr(func.body, 'body'):
                    final_expr = self._visit_expr(func.body.body)
                    if final_expr and final_expr != "None":
                        self._add_indented_line(f"return {final_expr}")
            else:
                # This might be a direct expression
                self._visit_binding_block(func.body)
        
        self.indent_level -= 1
    
    def _visit_binding_block(self, block: BindingBlock):
        """Visit a binding block and generate Python code."""
        if isinstance(block, DataflowBlock):
            # Dataflow blocks are converted to regular Python code
            for binding in block.bindings:
                self._visit_binding(binding)
        else:
            # Regular binding blocks
            for binding in block.bindings:
                self._visit_binding(binding)
    
    def _visit_binding(self, binding: Binding):
        """Visit a binding and generate corresponding Python code."""
        if isinstance(binding, VarBinding):
            self._visit_var_binding(binding)
        elif isinstance(binding, MatchCast):
            self._visit_match_cast(binding)
        elif isinstance(binding, If):
            self._visit_if_statement(binding)
    
    def _visit_var_binding(self, binding: VarBinding):
        """Visit a variable binding and generate assignment."""
        var_name = binding.var.name_hint
        value_expr = binding.value
        
        # Generate the right-hand side expression
        rhs_code = self._visit_expr(value_expr)
        
        # Add assignment statement
        self._add_indented_line(f"{var_name} = {rhs_code}")
    
    def _visit_expr(self, expr) -> str:
        """Visit an expression and generate Python code."""
        if isinstance(expr, Call):
            return self._visit_call(expr)
        elif isinstance(expr, Var):
            return expr.name_hint
        elif isinstance(expr, Constant):
            return self._visit_constant(expr)
        elif isinstance(expr, Tuple):
            return self._visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            return self._visit_tuple_get_item(expr)
        elif isinstance(expr, ShapeExpr):
            return self._visit_shape_expr(expr)
        elif isinstance(expr, PrimValue):
            return self._visit_prim_value(expr)
        else:
            # Fallback: use TVM's built-in printer
            return str(expr)
    
    def _visit_call(self, call: Call) -> str:
        """Visit a function call and generate Python code."""
        op = call.op
        
        # Handle different types of operations
        if hasattr(op, 'name'):
            op_name = op.name
            
            # Check if this is our custom call_py_func call disguised as call_tir
            # This check must come BEFORE checking op_mapping
            if self._is_call_py_func_disguised_as_call_tir(call):
                return self._generate_py_func_call(call)
            
            if op_name in self.op_mapping:
                # Map to PyTorch operation
                torch_op = self.op_mapping[op_name]
                args = [self._visit_expr(arg) for arg in call.args]
                
                # Handle special cases
                if torch_op == "self._call_tir_wrapper":
                    return self._generate_tir_call(call)
                elif torch_op == "self._call_dps_packed_wrapper":
                    return self._generate_dps_call(call)
                elif torch_op == "self._call_py_func_wrapper":
                    return self._generate_py_func_call(call)
                elif op_name == "relax.inspect.tensor_shape_i":
                    # Handle shape access: x.shape[0] -> x.shape[0]
                    if len(args) == 2:
                        tensor_expr = args[0]
                        axis_expr = args[1]
                        # Extract the axis value if it's a constant
                        if axis_expr.isdigit():
                            return f"{tensor_expr}.shape[{axis_expr}]"
                        else:
                            return f"{tensor_expr}.shape[{axis_expr}]"
                    else:
                        return self._generate_fallback_call(call)
                else:
                    # Regular PyTorch operation
                    if len(args) == 1:
                        return f"{torch_op}({args[0]})"
                    elif len(args) == 2:
                        return f"{torch_op}({args[0]}, {args[1]})"
                    else:
                        return f"{torch_op}({', '.join(args)})"
            else:
                # Unknown operation, use fallback
                return self._generate_fallback_call(call)
        else:
            # Variable or function call
            return self._generate_fallback_call(call)
    
    def _visit_constant(self, const: Constant) -> str:
        """Visit a constant and generate Python literal."""
        if hasattr(const, 'data'):
            data = const.data
            if hasattr(data, 'numpy'):
                numpy_data = data.numpy()
                if numpy_data.size == 1:
                    return str(numpy_data.item())
                else:
                    # Convert to PyTorch tensor
                    return f"torch.tensor({numpy_data.tolist()})"
        return "None"
    
    def _visit_tuple(self, tup: Tuple) -> str:
        """Visit a tuple and generate Python tuple."""
        elements = [self._visit_expr(elem) for elem in tup.fields]
        return f"({', '.join(elements)})"
    
    def _visit_tuple_get_item(self, get_item: TupleGetItem) -> str:
        """Visit a tuple get item and generate Python indexing."""
        tuple_expr = self._visit_expr(get_item.tuple_value)
        index = get_item.index
        if isinstance(index, int):
            return f"{tuple_expr}[{index}]"
        else:
            index_expr = self._visit_expr(index)
            return f"{tuple_expr}[{index_expr}]"
    
    def _visit_shape_expr(self, shape: ShapeExpr) -> str:
        """Visit a shape expression and generate Python shape."""
        values = []
        for val in shape.values:
            if hasattr(val, 'name_hint'):
                # This is a symbolic shape variable
                var_name = val.name_hint
                self.shape_vars[var_name] = True
                values.append(var_name)
            else:
                # This is a concrete value
                values.append(str(val))
        
        return f"({', '.join(values)})"
    
    def _extract_symbolic_shape(self, expr) -> str:
        """Extract symbolic shape expressions like x.shape[0]."""
        if hasattr(expr, 'name_hint'):
            return expr.name_hint
        elif hasattr(expr, 'value'):
            return str(expr.value)
        else:
            return str(expr)
    
    def _visit_prim_value(self, prim: PrimValue) -> str:
        """Visit a primitive value and generate Python literal."""
        value = prim.value
        if hasattr(value, 'value'):
            return str(value.value)
        else:
            return str(value)
    
    def _get_python_type_annotation(self, struct_info) -> str:
        """Convert Relax struct info to Python type annotation."""
        if isinstance(struct_info, TensorStructInfo):
            return "torch.Tensor"
        elif isinstance(struct_info, ShapeStructInfo):
            return "Tuple[int, ...]"
        elif isinstance(struct_info, PrimStructInfo):
            dtype = struct_info.dtype
            if dtype == "bool":
                return "bool"
            elif dtype.startswith("int"):
                return "int"
            elif dtype.startswith("float"):
                return "float"
            else:
                return "Any"
        elif isinstance(struct_info, TupleStructInfo):
            fields = [self._get_python_type_annotation(field) for field in struct_info.fields]
            return f"Tuple[{', '.join(fields)}]"
        elif isinstance(struct_info, ObjectStructInfo):
            return "Any"
        else:
            return "Any"
    
    def _generate_tir_call(self, call: Call) -> str:
        """Generate Python code for TIR function call."""
        # Extract TIR function name and arguments
        args = [self._visit_expr(arg) for arg in call.args]
        
        # For now, generate a placeholder
        return f"self._call_tir_wrapper({', '.join(args)})"
    
    def _generate_dps_call(self, call: Call) -> str:
        """Generate Python code for DPS packed function call."""
        # Extract function name and arguments
        args = [self._visit_expr(arg) for arg in call.args]
        
        # For now, generate a placeholder
        return f"self._call_dps_packed_wrapper({', '.join(args)})"
    
    def _generate_py_func_call(self, call: Call) -> str:
        """Generate Python code for Python function calls."""
        # Check if this is a Python function call disguised as call_tir
        # We look for GlobalVar with "__PYFUNC__" prefix in the first argument
        if (len(call.args) >= 2 and 
            hasattr(call.args[0], 'name_hint') and 
            isinstance(call.args[0].name_hint, str) and
            call.args[0].name_hint.startswith("__PYFUNC__")):
            
            # Extract function name from the GlobalVar name
            func_name = call.args[0].name_hint.replace("__PYFUNC__", "")
            
            # The second argument is a tuple containing the actual arguments
            if len(call.args) >= 2:
                args_tuple = call.args[1]
                if hasattr(args_tuple, 'fields'):
                    # Extract arguments from the tuple
                    remaining_args = [self._visit_expr(arg) for arg in args_tuple.fields]
                else:
                    remaining_args = []
            else:
                remaining_args = []
            
            # Generate the wrapper call
            if remaining_args:
                return f"self._call_py_func_wrapper('{func_name}', {', '.join(remaining_args)})"
            else:
                return f"self._call_py_func_wrapper('{func_name}')"
        else:
            # Not a Python function call, delegate to normal handling
            return self._visit_call_normal(call)
    
    def _visit_call_normal(self, call: Call) -> str:
        """Handle normal function calls (not Python function calls)."""
        op = call.op
        
        # Handle different types of operations
        if hasattr(op, 'name'):
            op_name = op.name
            if op_name in self.op_mapping:
                # Map to PyTorch operation
                torch_op = self.op_mapping[op_name]
                args = [self._visit_expr(arg) for arg in call.args]
                
                # Handle special cases
                if torch_op == "self._call_tir_wrapper":
                    return self._generate_tir_call(call)
                elif torch_op == "self._call_dps_packed_wrapper":
                    return self._generate_dps_call(call)
                elif torch_op == "self._call_py_func_wrapper":
                    return self._generate_py_func_call(call)
                elif self._is_call_py_func_disguised_as_call_tir(call):
                    # This is our custom call_py_func call disguised as call_tir
                    return self._generate_py_func_call(call)
                elif op_name == "relax.inspect.tensor_shape_i":
                    # Handle shape access: x.shape[0] -> x.shape[0]
                    if len(args) == 2:
                        tensor_expr = args[0]
                        axis_expr = args[1]
                        # Extract the axis value if it's a constant
                        if axis_expr.isdigit():
                            return f"{tensor_expr}.shape[{axis_expr}]"
                        else:
                            return f"{tensor_expr}.shape[{axis_expr}]"
                    else:
                        return self._generate_fallback_call(call)
                else:
                    # Regular PyTorch operation
                    if len(args) == 1:
                        return f"{torch_op}({args[0]})"
                    elif len(args) == 2:
                        return f"{torch_op}({args[0]}, {args[1]})"
                    else:
                        return f"{torch_op}({', '.join(args)})"
            else:
                return self._generate_fallback_call(call)
        else:
            return self._generate_fallback_call(call)
    
    def _is_call_py_func_disguised_as_call_tir(self, call: Call) -> bool:
        """Check if a call_tir call is actually a disguised call_py_func.
        
        We use call_tir as a base operator for call_py_func to avoid
        registration issues. This method detects such disguised calls.
        """
        # Check if this is a call_tir call
        if hasattr(call.op, 'name') and call.op.name == "relax.call_tir":
            # Check if the first argument starts with "__PYFUNC__"
            if len(call.args) > 0:
                first_arg = call.args[0]
                # Check if it's a GlobalVar with "__PYFUNC__" prefix
                if hasattr(first_arg, 'name_hint') and isinstance(first_arg.name_hint, str):
                    return first_arg.name_hint.startswith("__PYFUNC__")
                # Also check for PrimValue with "__PYFUNC__" prefix (fallback)
                elif hasattr(first_arg, 'value') and isinstance(first_arg.value, str):
                    return first_arg.value.startswith("__PYFUNC__")
        
        return False
    
    def _generate_fallback_call(self, call: Call) -> str:
        """Generate fallback Python code for unknown operations."""
        op = self._visit_expr(call.op)
        args = [self._visit_expr(arg) for arg in call.args]
        
        if len(args) == 0:
            return f"{op}()"
        else:
            return f"{op}({', '.join(args)})"
    
    def _add_indented_line(self, line: str):
        """Add an indented line to the Python code."""
        indent = "    " * self.indent_level
        self.python_code.append(f"{indent}{line}")
    
    def _has_return_statement(self, block: BindingBlock) -> bool:
        """Check if a binding block has a return statement."""
        # Simple check - in practice, we'd need more sophisticated analysis
        return False
    
    def _get_last_binding_var(self, block: BindingBlock) -> Optional[str]:
        """Get the variable name from the last binding."""
        if block.bindings:
            last_binding = block.bindings[-1]
            if isinstance(last_binding, VarBinding):
                return last_binding.var.name_hint
        return None


def print_relax_to_python(ir_mod: IRModule, config: Optional[PrinterConfig] = None) -> str:
    """Convert an IRModule containing Relax functions to Python code.
    
    Parameters
    ----------
    ir_mod : IRModule
        The IRModule to convert.
    config : PrinterConfig, optional
        Configuration for the printer.
        
    Returns
    -------
    str
        Generated Python code.
    """
    printer = RelaxToPythonPrinter()
    
    # Generate Python code for each Relax function
    python_functions = []
    
    for gv, func in ir_mod.functions_items():
        if isinstance(func, Function):
            func_name = gv.name_hint
            python_code = printer.print_relax_function(func, func_name)
            python_functions.append(python_code)
    
    # Combine all functions
    if python_functions:
        # Add imports
        imports = [
            "import torch",
            "import torch.nn.functional as F",
            "",
        ]
        
        # Add class definition for BasePyModule compatibility
        class_def = [
            "class RelaxToPythonModule:",
            "    \"\"\"Python module converted from Relax IRModule.\"\"\"",
            "    ",
            "    def __init__(self):",
            "        pass",
            "    ",
        ]
        
        # Add wrapper methods
        wrapper_methods = [
            "    def _call_tir_wrapper(self, *args):",
            "        \"\"\"Wrapper for TIR function calls.\"\"\"",
            "        # TODO: Implement TIR function calling",
            "        raise NotImplementedError(\"TIR function calling not yet implemented\")",
            "    ",
            "    def _call_dps_packed_wrapper(self, *args):",
            "        \"\"\"Wrapper for DPS packed function calls.\"\"\"",
            "        # TODO: Implement DPS function calling",
            "        raise NotImplementedError(\"DPS function calling not yet implemented\")",
            "    ",
            "    def _call_py_func_wrapper(self, func_name: str, *args):",
            "        \"\"\"Wrapper for Python function calls.\"\"\"",
            "        # TODO: Implement Python function calling",
            "        raise NotImplementedError(\"Python function calling not yet implemented\")",
            "    ",
        ]
        
        # Combine all parts
        all_code = imports + class_def + wrapper_methods + python_functions
        
        return "\n".join(all_code)
    else:
        return "# No Relax functions found in IRModule"


# Convenience function for direct usage
def relax_to_python(func: Function, func_name: str = None) -> str:
    """Convert a single Relax function to Python code.
    
    Parameters
    ----------
    func : Function
        The Relax function to convert.
    func_name : str, optional
        Name for the generated Python function.
        
    Returns
    -------
    str
        Generated Python code.
    """
    printer = RelaxToPythonPrinter()
    return printer.print_relax_function(func, func_name)
