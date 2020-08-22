# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""
A pass for analyzing device attribute of each IR node.
"""
from typing import Optional
from collections import defaultdict

import tvm
from ..expr_functor import ExprVisitor
from ..function import Function
from .. import ty, op, expr as _expr
from ... import register_func, cpu
from ..._ffi.runtime_ctypes import TVMContext

def is_closure(func):
    """Check if a function is a closure.

    Parameters
    ----------
    func : tvm.relay.Function
        The input function.

    Returns
    -------
        True if the input function is a closure, otherwise false.
    """
    return hasattr(func, 'attrs') and \
           hasattr(func.attrs, 'Closure') and int(func.attrs.Closure) == 1


def is_device_copy(call):
    """Check if a call node is a device copy call.

    Parameters
    ----------
    call : tvm.relay.Call
        The call node to be checked.

    Returns
    -------
    ret : Boolean
        True if the call is a device copy call. Otherwise, false.
    """
    if not isinstance(call, _expr.Call):
        return False
    if call.op == op.op.get("device_copy"):
        return True

    if not isinstance(call.op, Function):
        return False
    return isinstance(call.op.body, _expr.Call) and \
            call.op.body.op == op.op.get("device_copy")


class DeviceDomain:
    """A class to represent the device of a domain, i.e. a segment of relay
    program.

    Parameters
    ----------
    ctx : Optional[tvm.runtime.TVMContext]
        The device to be assigned to the current domain. It is optional.
    """
    def __init__(self, ctx: Optional[TVMContext]):
        self.domain = ctx

    def join(self, other: 'DeviceDomain') -> 'DeviceDomain':
        """Merge the device of two domains.

        Parameters
        ----------
        other : DeviceDomain
            The other domain to be merged.

        Returns
        -------
        ret : DeviceDomain
            The merged domain. An error will be raised if two domain has
            conflict, i.e. they have different context.
        """
        if self.domain is None and other.domain is None:
            return self
        elif self.domain is None:
            return other
        elif other.domain is None:
            return self
        elif (self.domain.device_type == other.domain.device_type and
              self.domain.device_id == other.domain.device_id):
            return self
        else:
            raise Exception("all expressions must have a singular device")

    def __hash__(self):
        if self.domain is None:
            return id(self)
        else:
            return hash((self.domain.device_type, self.domain.device_id))

    def __eq__(self, other):
        if self.domain is None and other.domain is None:
            return id(self) == id(other)
        else:
            return self.domain == other.domain


def bottom():
    """Create an empty domain. This would usually happen when we enter a new
    scope, i.e. Function.
    """
    return DeviceDomain(None)


def device_type(ctx):
    """Create a domain with the given device context.

    Parameters
    ----------
    ctx : tvm.runtime.TVMContext
        The device context used to construct a domain.

    Returns
    -------
    ret : DeviceDomain
        The constructed domain.
    """
    return DeviceDomain(ctx)


class ContextAnalysis(ExprVisitor):
    """Compute on which device each sub-expression will execute. A union find
    algorithm is used to assign and merge the context domains.

    Parameters
    ----------
    mod : tvm.IRModule
        The module that helps context analysis.

    current_func : tvm.relay.GlobalVar
        The current function that is being analyzed.

    fallback_device : tvm.runtime.TVMContext
        The default device that could be attached to an expression.
    """
    def __init__(self, mod, current_func, fallback_device):
        super().__init__()
        self.expr_to_device = defaultdict(bottom)
        self.device_uf = {}
        self.mod = mod
        self.closures = {}
        self.current_func = current_func
        self.fallback_device = fallback_device
        self.visited = {}

    def lookup(self, device):
        """Find the root domain of a given device domain.

        Parameters
        ----------
        device : DeviceDomain
            The domain that is used to query the root domain.

        Returns
        -------
        ret : DeviceDomain
            The root domain.
        """
        while device in self.device_uf:
            device = self.device_uf[device]
        return device

    def unify(self, lhs, rhs):
        """Unify the device context of two domains.

        Parameters
        ----------
        lhs : DeviceDomain
            The lhs domain to unify.

        rhs : DeviceDomain
            The rhs domain to unify.

        Returns
        -------
        ret : DeviceDomain
            The unified domain.
        """
        lhs = self.lookup(lhs)
        rhs = self.lookup(rhs)
        unified_device = lhs.join(rhs)
        if not lhs == unified_device:
            self.device_uf[lhs] = unified_device
        if not rhs == unified_device:
            self.device_uf[rhs] = unified_device
        return unified_device

    def unify_expr(self, lhs, rhs):
        """Compute the device type of both expressions and unify them.

        Parameters
        ----------
        lhs : tvm.relay.Expr
            The lhs expression to unify.

        rhs : tvm.relay.Expr
            The rhs expression to unify.

        Returns
        -------
        ret : DeviceDomain
            The unified domain.
        """
        return self.unify(self.device_for(lhs), self.device_for(rhs))

    def device_for(self, expr):
        """Find the domain that contains the given expr.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression used to lookup a domain.

        Returns
        -------
        ret : DeviceDomain
            The domain that contains the expression.
        """
        return self.lookup(self.expr_to_device[expr])

    def device_copy(self, inps, outputs, src_dev_type, dst_dev_type):
        """Unify the device context for device copy node. Device copy node is
        the only node that carries information in the input program. The device
        attribute of other nodes are propagated from it.

        Parameters
        ----------
        inps : List[tvm.relay.Expr]
            The input expression to the device copy node. The device type of
            the input should be the same as the source device type of the
            copy node.

        outputs : List[tvm.relay.Expr]
            The output expression of the device copy node. The device type of
            the output should be the same as the destination device type of the
            copy node.

        src_dev_type : int
            The source device type of the copy node.

        dst_dev_type : int
            The destination device type of the copy node.
        """
        src_dev_type = device_type(TVMContext(src_dev_type, 0))
        for inp in inps:
            self.unify(self.device_for(inp), src_dev_type)

        dst_dev_type = device_type(TVMContext(dst_dev_type, 0))
        for output in outputs:
            self.unify(self.device_for(output), dst_dev_type)

    def unify_call(self, call_op, inputs, outputs, device=None):
        """Unify the domain of inputs and outputs of a relay Call.

        Parameters
        ----------
        op : tvm.relay.Expr
            The op of a call node.

        inputs : List[tvm.relay.Expr]
            The inputs of the call.

        outputs : List[tvm.relay.Expr]
            The outputs of the call.

        Returns
        -------
            The unified domain.

        Note
        ----
        For most call nodes, the op, inputs, and outputs should all be in the
        same domain, i.e. having the same context. However, device_copy call node
        needs to be handled differently as it copies data from one device to
        another.
        """
        device = device if device else bottom()
        for arg in inputs:
            device = self.unify(device, self.device_for(arg))

        device = self.unify(device, self.device_for(call_op))

        for out in outputs:
            device = self.unify(device, self.device_for(out))

        return device

    def visit_call(self, call):
        if is_device_copy(call):
            inps = [call.args[0]]
            outs = [call]
            if isinstance(call.op, Function):
                # device_copy is fused, propagate device to the fused function
                inps.append(call.op.params[0])
                outs.append(call.op)
                body = call.op.body
                assert isinstance(body, _expr.Call) and is_device_copy(body)
                outs.append(call.op.body)
                src_dev_type = call.op.body.attrs.src_dev_type
                dst_dev_type = call.op.body.attrs.dst_dev_type
            else:
                src_dev_type = call.attrs.src_dev_type
                dst_dev_type = call.attrs.dst_dev_type

           # Device copy op only has one input which is now annotated with the
           # same device to the source device type of the device copy op.
           # The call itself has the same device type to the destination.
            self.device_copy(inps, outs, src_dev_type, dst_dev_type)
            super().visit_call(call)
        elif call.op == op.op.get("memory.alloc_storage"):
            # The arguments should be on CPU.
            for arg in (call.args[0], call.args[1]):
                self.unify(self.device_for(arg), device_type(cpu(0)))
                self.visit(arg)
            call_dev = device_type(TVMContext(call.attrs.device_type,
                                              call.attrs.device_id))
            self.unify(self.device_for(call), call_dev)
        elif call.op == op.op.get("memory.alloc_tensor"):
            storage, shape = call.args[0], call.args[1]
            self.unify(self.device_for(storage), self.device_for(call))
            # The shape for alloc_tensor should be on CPU.
            self.unify(self.device_for(shape), device_type(cpu(0)))
            self.visit(shape)
        elif call.op == op.op.get("vm.shape_func"):
            shape_func_domain = device_type(cpu(0))
            # No need to union the op of a shape_func as shape_func doesn't
            # invoke the op itself. It should be handled by invoke_tvm_op.
            # Therefore, we skip call.args[0] here.
            self.unify_call(call, call.args[1].fields,
                            call.args[2].fields, shape_func_domain)
            for arg in call.args[1]:
                self.visit(arg)
            for arg in call.args[2]:
                self.visit(arg)
        elif call.op == op.op.get("vm.invoke_tvm_op"):
            device = self.unify_call(call.args[0], call.args[1].fields,
                                     call.args[2].fields)
            self.unify(self.device_for(call), device)
            super().visit_call(call)
        elif call.op == op.op.get("vm.shape_of"):
            # vm shape_of is always on the CPU.
            self.visit(call.args[0])
            self.unify(self.device_for(call), device_type(cpu(0)))
        elif call.op == op.op.get("vm.reshape_tensor"):
            data, shape = call.args
            self.unify(self.device_for(call), self.device_for(data))
            # The shape field of reshape_tensor is always on the CPU.
            self.unify(self.device_for(shape), device_type(cpu(0)))
            self.visit(data)
            self.visit(shape)
        elif isinstance(call.op, Function):
            device = self.device_for(call)
            for arg in call.args:
                device = self.unify(device, self.device_for(arg))
                self.visit(arg)

            for param in call.op.params:
                self.visit(param)
                device = self.unify(device, self.device_for(param))

            self.unify(device, self.device_for(call.op))
            self.unify(device, self.device_for(call.op.body))
            # self.visit(call.op)
        elif isinstance(call.op, _expr.GlobalVar):
            device = self.device_for(call)
            assert self.mod, "Cannot analyze context on a globalvar without module"
            func = self.mod[call.op]

            assert len(call.args) == len(func.params)

            for arg, param in zip(call.args, func.params):
                self.visit(arg)
                # Save the the arg to function mapping for closures as it will
                # be invoked/unified later.
                if isinstance(arg.checked_type, ty.FuncType):
                    if arg in self.closures:
                        self.closures[param] = self.closures[arg]
                    else:
                        assert isinstance(arg, _expr.GlobalVar)
                        self.closures[param] = arg
                self.unify(self.device_for(arg), self.device_for(param))

            device = self.unify(device, self.device_for(call.op))
            device = self.unify(device, self.device_for(func))
            device = self.unify(device, self.device_for(func.body))
            # Step into the callee. We need to skip recursive calls, otherwise, it
            # would be a infinite loop, so does mutual recursive calls
            cur_func = self.current_func
            self.current_func = call.op
            if cur_func.name_hint != call.op.name_hint:
                self.visit(func)
                self.visited[func] = device
            self.current_func = cur_func
        elif isinstance(call.op, _expr.Var):
            # It is a closure when we call a var
            # Unify the corresponding arguement and parameter
            device = self.device_for(call)
            assert call.op in self.closures, f"Cannot find {call.op}"
            glb_var = self.closures[call.op]
            assert self.mod, "Cannot analyze context on a globalvar without module"
            func = self.mod[glb_var]
            # Unify the underlying function for clousre or currying funcitons.
            while is_closure(func) or (isinstance(func.body, _expr.Let) and
                                       func.body.var in self.closures):
                device = self.unify(device, self.device_for(func))
                if is_closure(func):
                    func = func.body
                elif (isinstance(func.body, _expr.Let) and func.body.var in self.closures):
                    func = self.mod[self.closures[func.body.var]]

            assert isinstance(func, Function)
            assert len(call.args) == len(func.params)

            for dev_arg, dev_param in zip(call.args, func.params):
                self.visit(dev_arg)
                self.unify(self.device_for(dev_arg), self.device_for(dev_param))

            device = self.unify(device, self.device_for(call.op))
            device = self.unify(device, self.device_for(glb_var))
            device = self.unify(device, self.device_for(func))
            cur_func = self.current_func
            # Step into the closure.
            self.current_func = glb_var
            if not tvm.ir.structural_equal(cur_func, glb_var):
                self.visit(func)
                self.visited[func] = device
            self.current_func = cur_func
        else:
            self.unify_call(call, call.args, [call])
            super().visit_call(call)

    def _extract_closure(self, expr):
        while isinstance(expr, _expr.Let):
            expr = expr.value
            if isinstance(expr, _expr.GlobalVar):
                return expr
            elif isinstance(expr, _expr.Call) and isinstance(expr.op,
                                                             _expr.GlobalVar):
                return expr.op
        return None

    def visit_let(self, let):
        while isinstance(let, _expr.Let):
            # Save currying/closures since they will be invoked later
            if isinstance(let.value.checked_type, ty.FuncType):
                gv = self._extract_closure(let)
                assert gv
                self.closures[let.var] = gv

            self.unify(self.device_for(let.var), self.device_for(let.value))
            self.unify_expr(let, let.body)
            self.visit(let.value)
            let = let.body

        self.visit(let)

    def visit_function(self, f):
        if f in self.visited and self.visited[f].domain:
            return

        device = self.unify(self.device_for(f), self.device_for(f.body))
        for x in f.params:
            self.device_for(x)
        self.visit(f.body)
        self.visited[f] = device

    def visit_tuple(self, tup):
        # We only support tuple with the same of device.
        if not tup:
            return
        device = self.device_for(tup[0])
        for i in range(1, len(tup)):
            device = self.unify(device, self.device_for(tup[i]))
        self.unify(device, self.device_for(tup))
        super().visit_tuple(tup)

    def visit_tuple_getitem(self, t):
        value = t.tuple_value
        if isinstance(t.tuple_value, _expr.Tuple):
            self.unify(self.device_for(t), self.device_for(value))
            value = t.tuple_value[t.index]
        self.unify(self.device_for(t), self.device_for(value))
        super().visit_tuple_getitem(t)

    def visit_match(self, m):
        # For match node, we unify the value and the rhs of each clause
        device = self.unify(self.device_for(m), self.device_for(m.data))
        for c in m.clauses:
            device = self.unify(device, self.device_for(c.rhs))
        super().visit_match(m)

    def visit_global_var(self, gv):
        self.device_for(gv)

    def visit_var(self, var):
        self.device_for(var)

    def visit_constant(self, const):
        self.device_for(const)

    def results(self):
        """Return the analysis result.

        Returns
        -------
        ret : Dict[tvm.relay.Expr, DeviceDomain]
            The dictionary mapping each expression to a device context.
        """
        results = {}
        for exp in self.expr_to_device:
            device = self.lookup(self.expr_to_device[exp])
            if device.domain is None:
                results[exp] = self.fallback_device
            else:
                results[exp] = device.domain

        return results


def mk_analysis_annotator(results):
    """Pretty print the annotated relay program with device info"""
    def _annotator(exp):
        if exp in results:
            return f"<{results[exp]}>"
        else:
            return ""

    return _annotator


def context_analysis(mod, fallback_device):
    """Perform device context analysis on a given relay program. This requires
    that the program has already been annotated and rewritten by replacing on
    device annotations with device copy nodes.

    Parameters
    ----------
    mod : tvm.IRModule
        The IRModule for analysis

    fallback_device : tvm.runtime.TVMContext
        The default device context

    Returns
    -------
    ret : Dict[tvm.relay.Expr, [int]]
        The mapping of each expression to the device context that is
        represented in a list form as TVMContext is not a runtime object.
    """
    assert isinstance(mod, tvm.IRModule)
    # TODO(@zhiics) Apply the pass to all functions/entries
    entry = mod.get_global_var("main")
    ca = ContextAnalysis(mod, entry, fallback_device)
    expr = mod[entry]
    ca.visit(expr)
    ret = defaultdict(list)
    for key, val in ca.results().items():
        ret[key] = [val.device_type, val.device_id]
    return ret


register_func("relay.analysis.ContextAnalysis", context_analysis)
