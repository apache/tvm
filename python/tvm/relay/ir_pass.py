# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
The optimizer for Relay.

Exposes an interface for configuring the optimizer and scripting
it directly in Python.
"""
from typing import TypeVar, Generic, Union
from typing import Dict, Tuple, List, Callable
import tvm

from .expr import Expr
from .expr import Function, Let, Call, LocalVar
from .expr import GlobalVar, If, Constant
from .type import Type, TypeParam
from .env import Environment
from .op import Op
from .op.op import specialize_op
# import relay.make as relay_mk
# from relay import ir
# from relay.env import Environment
# from relay.tyck import check_expr
# from relay.first_order_reverse_ad import fo_with_gradient
# from relay.anf import to_anf
from . import _ir_pass

# Expose checking expression, should rename to infer_type.
check_expr = _ir_pass.check_expr

# # pylint: disable=invalid-name
# concretize = _opt.concretize

# # pylint: disable=invalid-name
# optimize = _opt.optimize

# # pylint: disable=invalid-name
# type_specialize = _opt.type_specialize

# # pylint: disable=invalid-name
# compile_ops_to_module = _opt.compile_ops_to_module


@tvm.register_func("relay.mangle")
def mangle(name: str, types: List[Type]) -> str:
    for typ in types:
        name += str(typ) + "_"
    return name

T = TypeVar('T')
class AbstractExprVisitor(Generic[T]):
    """A functional visitor over Expr in Python."""

    # pylint: disable=no-else-return
    def visit(self, expr: Expr) -> T:
        """Apply the visitor to an expression."""
        if isinstance(expr, Function):
            return self.visit_function(expr)
        elif isinstance(expr, Call):
            return self.visit_call(expr)
        elif isinstance(expr, Let):
            return self.visit_let(expr)
        elif isinstance(expr, LocalVar):
            return self.visit_local_var(expr)
        elif isinstance(expr, GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, If):
            return self.visit_if(expr)
        elif isinstance(expr, Tuple):
            return self.visit_tuple(expr)
        elif isinstance(expr, Constant):
            return self.visit_constant(expr)
        else:
            raise Exception(f"warning unhandled case: {type(expr)}")

    def visit_function(self, _: Function) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_let(self, _: Let) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_call(self, _: Call) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_local_id(self, _: LocalVar) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_type(self, typ: Type) -> Type:
        return typ

    def visit_if(self, _: If) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_tuple(self, _: Tuple) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_constant(self, _: Constant) -> T:
        raise Exception("Abstract method please implement me.")

    def visit_global_var(self, _: GlobalVar) -> T:
        raise Exception("Abstract method please implement me.")

    @classmethod
    def to_pass(cls) -> Callable[[Environment], Callable[[GlobalVar, Function], Function]]:
        def _outer_wrapper(env):
            visitor = cls(env)
            def _inner_wrapper(var, func):
                return visitor.visit(func)
            return _inner_wrapper
        return _outer_wrapper

class ExprVisitor(AbstractExprVisitor[Expr]):
    """A functional visitor over Expr in Python."""

    def visit_function(self, fn: Function) -> Expr:
        new_body = self.visit(fn.body)
        return Function(
            list(fn.params),
            fn.ret_type, new_body,
            fn.type_params)

    def visit_let(self, let: Let) -> Expr:
        new_var = self.visit(let.var)
        new_value_type = self.visit_type(let.value_type)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return Let(new_var, new_val, new_body, new_value_type)

    def visit_call(self, call: Call) -> Expr:
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_fn, new_args, call.attrs)

    def visit_local_var(self, local_var: LocalVar) -> Expr:
        return local_var

    def visit_global_id(self, global_var: GlobalVar) -> Expr:
        return global_var

    def visit_if(self, ite: If) -> Expr:
        return If(
            self.visit(ite.guard),
            self.visit(ite.true_b),
            self.visit(ite.false_b))

    def visit_tuple(self, tup: Tuple) -> Expr:
        return Tuple([self.visit(field) for field in tup.fields])

    def visit_constant(self, const: Constant) -> Expr:
        return const

MMCacheKey = Tuple[Union[GlobalVar, str], List[Type]]

class Monomorphize(ExprVisitor):
    """A monomorphization pass.

       Implements what is known as "monomorphization" in
       classic compiler literature. This pass removes
       polymorphism replacing calls to functions and
       operators with type specialized versions.
    """
    monomorph_map: Dict[MMCacheKey, Union[Op, Function]]

    # pylint: disable=super-init-not-called
    def __init__(self, env: Environment) -> None:
        self.env = env
        # Stores (GlobalVar, Type), should eventually store attributes.
        self.monomorph_map = {}

    # pylint: disable=no-else-return
    def visit_call(self, call: Call) -> Expr:
        cache_key = (call.op, call.type_args)
        new_args = [self.visit(arg) for arg in call.args]

        if cache_key in self.monomorph_map:
            op = self.monomorph_map[cache_key]
            new_args = [self.visit(arg) for arg in call.args]
            return Call(op, new_args, call.attrs)
        else:
            if isinstance(call.op, Op):
                poly_name = call.op.name
                mono_name = mangle(poly_name, call.type_args)
                for arg in call.type_args:
                    if isinstance(arg, TypeParam):
                        return call # raise Exception("...") # Fix me in the morning!!!

                mono_op = specialize_op(poly_name, mono_name, call.type_args)
                self.monomorph_map[cache_key] = mono_op
                return Call(mono_op, new_args,call.attrs, [])
            elif isinstance(call.op, GlobalVar):
                return call
                # defn = self.env.lookup(call.op)
                # new_id = self.env.global_id(defn.id.name + str(1))
                # cache_key = (call.op, call.type_args)
                # self.monomorph_map[cache_key] = new_id
                # new_body = self.visit(type_specialize(call.type_args, defn.body))
                # new_body = Function(
                #     [], new_body.params, new_body.ret_type, new_body.body)
                # new_ty = check_expr(self.env, new_body)
                # # TODO(@jroesch): move into C++
                # # TODO(@joresch): implement and call name mangler
                # defn = Defn(new_id, new_ty, new_body)
                # self.env.add(defn)
                # self.visit_item(defn)
                # return Call(new_id, call.args, call.attrs)
                
            elif isinstance(call.op, Function):
                return call
                # new_func = type_specialize(call.type_args, call.op)
                # new_func = self.visit(new_func)
                # new_func = Function([],
                #                          new_func.params,
                #                          new_func.ret_type,
                #                          new_func.body)
                # check_expr(self.env, new_func)
                # return Call(new_func, call.args, call.attrs)
            else:
                new_fn = self.visit(call.op)
                return Call(new_fn, new_args, call.attrs)


# TODO(@jroesch): Fix up my type
__tgt_host__ = __tgt__ = "llvm"
__relay_tvm_context__ = tvm.cpu()

