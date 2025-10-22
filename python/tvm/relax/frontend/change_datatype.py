from operator import call
import tvm
from tvm import relax
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.expr import Constant, Var, DataflowVar, Function, VarBinding, DataflowBlock, Call
from tvm.relax.struct_info import TensorStructInfo
from tvm import tir
from tvm.relax.expr import Tuple

@mutator
class ChangeDatatype(PyExprMutator):
    def __init__(self, src: str, dst: str, mod=None) -> None:
        super().__init__()
        self.src = src
        self.dst = dst

    def visit_constant_(self, const: Constant) -> relax.Expr:
        if const.data.dtype == self.src:
            return relax.op.astype(const, self.dst)
        return const

    def visit_var_def_(self, var: Var) -> Var:
        si = var.struct_info
        if isinstance(si, TensorStructInfo) and si.dtype == self.src:
            new_si = TensorStructInfo(si.shape, self.dst)
            new_var = Var(var.name_hint, new_si)
            self.set_var_remap(var.vid, new_var)
            return new_var
        return var

    def visit_dataflow_var_def_(self, var: DataflowVar) -> DataflowVar:
        si = var.struct_info
        if isinstance(si, TensorStructInfo) and si.dtype == self.src:
            new_si = TensorStructInfo(si.shape, self.dst)
            new_var = DataflowVar(var.name_hint, new_si)
            self.set_var_remap(var.vid, new_var)
            return new_var
        return var

    def visit_var_(self, var: Var) -> Var:
        remapped = self.get_var_remap(var.vid)
        return remapped if remapped is not None else var

    def visit_function_(self, fn: Function) -> Function:
        new_params = [self.visit_var_def(p) for p in fn.params]
        new_body = self.visit_expr(fn.body)
        ret_si = fn.ret_struct_info
        if isinstance(ret_si, TensorStructInfo) and ret_si.dtype == self.src:
            new_ret = TensorStructInfo(ret_si.shape, self.dst)
        else:
            new_ret = ret_si
        return Function(new_params, new_body, ret_struct_info=new_ret, attrs=fn.attrs)

    def visit_call_(self, call: Call) -> Call:
        new_args = [self.visit_expr(arg) for arg in call.args]
        # attrs = {} if call.attrs is None else dict(call.attrs)
        attrs = {}
        if call.attrs is not None:
            attrs = {
                k: getattr(call.attrs, k)
                for k in dir(call.attrs)
                if not k.startswith("_") and not callable(getattr(call.attrs, k))
            }
        # if call.op.name == "relax.nn.max_pool2d":
        #     # 將輸入轉換為custom[posit32]32
        #     posit_args = [relax.op.astype(arg, "custom[posit32]32") for arg in new_args]
        #     # 使用posit32進行計算
        #     result = relax.op.nn.max_pool2d(*posit_args, **attrs)
        #     return relax.op.astype(result, self.dst)
        
        # if call.op.name == "relax.nn.softmax":
        #     # 將輸入轉換為custom[posit32]32
        #     posit_args = [relax.op.astype(arg, "custom[posit32]32") for arg in new_args]
        #     # 使用posit32進行計算
        #     result = relax.op.nn.softmax(*posit_args, **attrs)
        #     return relax.op.astype(result, self.dst)
            
        if call.op.name in "relax.sqrt":
            out = relax.Call(call.op, new_args, call.attrs)
            return relax.op.astype(out, self.dst)

        if call.op.name == "relax.astype":
            orig = attrs.get("dtype")
            new_dtype = self.dst if orig == self.src else orig
            return relax.op.astype(new_args[0], new_dtype)

        if attrs.get("out_dtype") == self.src:
            attrs["out_dtype"] = self.dst

        if call.op.name == "relax.nn.conv2d":
            return relax.op.nn.conv2d(*new_args, **attrs)

        if call.op.name == "relax.matmul":
            return relax.op.matmul(*new_args, **attrs)
        
        return Call(call.op, new_args, call.attrs, call.span)
