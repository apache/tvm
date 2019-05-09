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
"""Utility for converting Relay code into a Python script with equivalent semantics"""
import ast
import re

import astor
import numpy
from tvm import relay
from relay.expr import Expr
from relay.expr_functor import ExprFunctor

INTERPRETER_VAR = '_INTRP'
OUTPUT_VAR_NAME = '_py_out'

PROLOGUE = '''
import tvm
from tvm import relay
from relay.backend.interpreter import RefValue, TupleValue, TensorValue
from relay import create_executor

{} = create_executor()

'''.format(INTREPRETER_VAR)

class PythonConverter(ExprFunctor):
    '''Functor for translating Relay programs into Python ASTs.'''

    def __init__(self, mod) -> None:
        super().__init__()
        self.mod = mod
        self.fun_no = 0
        self.var_no = 0
        self.var_map = {}


    def convert(self, prog: Expr):
        '''This method converts the passed Relay expression into a Python
        AST object with equivalent semantics.

        The Python AST can be executed using exec(); it can be turned
        into text and inspected using astor.
        '''
        check = relay.ir_pass.infer_type(prog, self.mod)
        assert relay.ir_pass.well_formed(check)

        # start with conversion prelude (imports) and convert global defs
        start = ast.parse(PROLOGUE)
        body = start.body # build up a body as a list of statements
        body += self.convert_module()

        prog_body, extra_defs = self.visit(check)
        body += extra_defs

        # we finally must assign the final expression to the output var
        # so it can be read after running EXEC
        body.append(ast.Assign(ast.Name(OUTPUT_VAR_NAME, ast.Store()), prog_body))

        return ast.Module(body=body)


    # Only checks that the name does not contain invalid characters
    # (underscores, numbers, and letters are permitted). Since
    # we append a number and underscore to var names anyway, it doesn't
    # matter if the name is the empty string
    def check_safe_name(self, name: str):
        return re.match(r'\w*$', name)


    # generates a unique variable name starting from the hint
    def generate_var_name(self, name_hint: str):
        if not check_safe_name(name_hint):
            raise Exception('Name hint contains invalid characters: {}'.format(name_hint))
        name = '{}_var_{}'.format(name_hint, self.var_no)
        self.var_no += 1
        return name


    # generates a unique function name starting from the hint
    def generate_function_name(self, name_hint: str):
        if not check_safe_name(name_hint):
            raise Exception('Name hint contains invalid characters: {}'.format(name_hint))
        name = '{}_fun_{}'.format(name_hint, self.fun_no)
        self.fun_no += 1
        return name


    # returns the var name for the given Relay variable
    def get_var_name(self, var: Expr):
        if var in self.var_map:
            return self.var_map[var]
        name = self.generate_var_name(var.name_hint)
        self.var_map[var] = name
        return name


    # returns an variable AST node for the given Relay var depending on
    # whether it must appear in an assignment or not
    def include_var(self, var: Expr, assign=False):
        name = self.get_var_name(var)
        return ast.Name(id=name, ctx=ast.Store() if assign else ast.Load())


    # Given a list of call args or tuple fields, converts each
    # and returns their ASTs and their defs lists (in order)
    def convert_fields(self, fields: [Expr]):
        bodies = []
        defs = []
        for field in fields:
            member_body, member_defs = self.visit(field)
            bodies.append(member_body)
            defs += member_defs
        return (bodies, defs)


    # wraps the passed expression in a thunk
    def convert_to_thunk(self, name_hint: str, expr: Expr):
        body, defs = self.visit(expr)
        thunk_name = self.generate_function_name(name_hint)
        thunk = ast.FunctionDef(thunk_name,
                                ast.arguments([]),
                                defs + [ast.Return(body)])
        return (thunk, thunk_name)


    # converts the given Relay function into a Python function
    def convert_func_node(self, name_hint: str, func: Expr):
        func_name = self.generate_function_name(name_hint)
        var_names = [self.get_var_name(var) for var in func.params]
        body, defs = self.visit(func.body)
        ret = ast.FunctionDef(func_name, ast.arguments(var_names),
                              defs + [ast.Return(body)])
        return (ret, func_name)


    # converts all the global functions defined in the module and returns
    # them as a list of definitions
    def convert_module(self):
        defs = []
        for var, func in self.mod.functions.items():
            converted_func, func_name = self.convert_func_node(var.name_hint, func)
            defs.append(converted_func)
            # need to add this assignment so references to the global var in the program
            # go to the function!
            defs.append(self.parse_assignment('{} = {}'.format(var.name_hint, func_name)))
        return defs


    # parses a string of python code corresponding to an expression (not a statement)
    # into an AST object, returns the expression (not as a statement)
    def parse_single_expression(self, code: str):
        parsed = ast.parse(code)
        assert len(parsed.body) == 1
        # ast.Expr is an "expression statement," but we want the inner expr
        assert isinstance(parsed.body[0], ast.Expr)
        return parsed.body[0].value


    # parses a python assignment statement into an AST object
    def parse_assignment(self, code: str):
        parsed = ast.parse(code)
        assert len(parsed.body) == 1
        assert isinstance(parsed.body[0], ast.Assign)
        return parsed.body[0]


    # simple function call
    def create_call(self, func_name: str, arguments):
        return ast.Call(ast.Name(func_name, ctx=ast.Load()), arguments, [])


    def create_op_call(self, op: Expr, num_args: int, attrs):
        '''Wraps a call to an operator with an invocation of the interpreter,
        like in the tests. This is pretty dirty but is the simplest way to
        invoke operators from Python code'''

        arg_names = [self.generate_var_name('_{}_arg_{}'.format(op.name, i))
                     for i in range(num_args)]
        var_names = [self.generate_var_name('_{}_var_{}'.format(op.name, i))
                     for i in range(num_args)]
        call_name = self.generate_var_name('_{}_call'.format(op.name))

        var_assignments = ['{} = relay.var(\'{}\')'.format(name, name) for name in var_names]
        body = [self.parse_assignment(a) for a in var_assignments]
        assert len(body) == num_args

        # call = relay.op(relay_vars, attr=value)
        call_args = ', '.join(var_names)
        if attrs is not None:
            attr_assignments = ['{}={}'.format(key, repr(attrs[key])) for key in attrs.keys()]
            call_args += ', ' + ', '.join(attr_assignments)
        call_assignment = '{} = relay.{}({})'.format(call_name, op.name, call_args)
        body.append(self.parse_assignment(call_assignment))

        # return _INTRP.evaluate(call, { relay_var : argument })
        arg_assignments = ['{} : {}'.format(arg_names[i], var_names[i])
                           for i in range(num_args)]
        arg_dict = '{{ {} }}'.format(', '.join(arg_assignments))
        intrp_call = '{}.evaluate({}, {})'.format(INTERPRETER_VAR, call_name, arg_dict)
        body.append(ast.Return(self.parse_single_expression(intrp_call)))

        func_name = self.generate_function_name('_op_call_{}'.format(op_name))
        func = ast.FunctionDef(func_name, ast.arguments(arg_names), body)
        return (func, func_name)

    # Convention for the expr visitor: Each visit function returns a tuple of two members.
    #
    # The first is a Python AST comprised of a single *expression* that evaluates to an equivalent
    # result to the desired Relay expression (and executes all effects in the right order).
    #
    # The second is a list of function definition *statements* defining thunks and other
    # auxiliary functions needed in the translated AST object. The defs in the second object
    # will always have unique names and will never perform any effects, so as long as they
    # appear in the Python program before the first statement is executed, there should not
    # be any problems.

    def visit_var(self, var: Expr):
        return (self.include_var(var, assign=False), [])


    def visit_global_var(self, gvar: Expr):
        # we don't need to add numbers to global var names because
        # the *names* are checked for uniqueness in the mod
        return (ast.Name(id=gvar.name_hint, ctx=ast.Load()), [])


    def visit_let(self, letexp: Expr):
        # To properly account for scoping and ensure that the entire node produces an expression,
        # we translate the let binding as a function that we call with the value we intend to bind.
        # Yes, this is somewhat ugly.
        '''
        let var = body in value
        =======================
        def let_thunk(var):
            return value
        let_thunk(body)
        '''
        value_body, value_defs = self.visit(letexp.value)

        func_name = self.generate_function_name('_let_func')
        binding_func = ast.FunctionDef(func_name, ast.arguments([self.get_var_name(letexp.var)]),
                                       value_defs + [ast.Return(value_body)])

        # we call the binding func with the intended value for the bound variable
        bind_body, bind_defs = self.visit(letexp.body)
        bind_defs.append(binding_func)
        binding_call = self.create_call(func_name, [bind_body])

        return (binding_call, bind_defs)


    def visit_tuple(self, tup: Expr):
        fields, ret_defs = self.convert_fields(tup.fields)
        return (self.create_call('TupleValue', fields), ret_defs)


    def visit_tuple_getitem(self, tgi: Expr):
        tup, tup_defs = self.visit(tgi.tuple_value)
        ret = ast.Subscript(value=tup, slice=ast.Index(value=ast.Num(n=tgi.index)), ctx=ast.Load())
        return (ret, tup_defs)


    def visit_if(self, if_block: Expr):
        cond_body, cond_defs = self.visit(if_block.cond)
        true_body, true_defs = self.visit(if_block.true_branch)
        false_body, false_defs = self.visit(if_block.false_branch)
        ret = ast.IfExp(cond_body, true_body, false_body)
        return (ret, cond_defs + true_defs + false_defs)


    def visit_constant(self, constant: Expr):
        value = constant.data.asnumpy()
        arr_literal = 'numpy.array({})'.format(
            numpy.array2string(value, separator=','))
        const_expr = self.parse_single_expression(arr_literal)
        return (self.create_call('TensorValue', [const_expr]), [])


    def visit_function(self, func: Expr):
        # Python's lambdas are very restrictive, so we do "name" inline functions
        converted_func, func_name = self.convert_func_node('_anon_func', func)
        return (self.include_var(func_name), [converted_func])


    def visit_call(self, call: Expr):
        '''For calls, we must distinguish between ordinary functions,
        operators, and constructor calls.'''
        func = call.op
        fields, field_defs = self.convert_fields(call.args)
        if isinstance(func, relay.Op):
            op_func, op_call = self.create_op_call(func, len(call.args), call.attrs)
            defs = [op_func] + field_defs
            return (ast.Call(ast.Name(op_call, ctx=ast.Load()), fields, []), defs)

        if isinstance(func, relay.Constructor):
            raise Exception('ADTs not supported')

        # ordinary function
        converted_func, defs = self.visit(func)
        defs += field_defs
        return (ast.Call(converted_func, fields, []), defs)


    def visit_ref_create(self, ref: Expr):
        val, defs = self.visit(ref.value)
        return (self.create_call('RefValue', [val]), defs)


    def visit_ref_read(self, read: Expr):
        ref, defs = self.visit(read.ref)
        return (ast.Attribute(ref, attr='value', ctx=ast.Load()), defs)


    def visit_ref_write(self, write: Expr):
        '''For writing refs, we wrap the update in a thunk
        (returning an empty tuple to match Relay's semantics)
        that we execute at the right time. This ensures such assignments
        can be properly nested, since assignments are statements
        in Python but expressions in Relay'''
        ref, ref_defs = self.visit(write.ref)
        val, val_defs = self.visit(write.value)
        thunk_name = self.generate_function_name('_ref_write_thunk')
        thunk = ast.FunctionDef(thunk_name,
                                ast.arguments([]),
                                ref_defs + val_defs + [
                                    ast.Assign(
                                        ast.Attribute(ref, attr='value', ctx=ast.Store()),
                                        val),
                                    ast.Return(self.create_call('TupleValue', []))
                                ])
        return (self.create_call(thunk_name, []), [thunk])


    def visit_match(self, _):
        raise Exception('Pattern matching not yet suported')


def to_python(expr: Expr, mod=relay.Module()) -> str:
    '''Converts the given Relay expression into a Python script.'''
    converter = PythonConverter(mod)
    py_ast = converter.convert(expr)
    return astor.to_source(py_ast)


def run_as_python(expr: Expr, mod=relay.Module()):
    '''Converts the given Relay expression into a Python script and
    executes it.'''
    py_ast = to_python_ast(expr, mod)
    code = compile(py_ast, '<string>', 'exec')
    output_map = {OUTPUT_VAR_NAME : None}
    #pylint: disable=exec-used
    exec(code, {}, output_map)
    return output_map[OUTPUT_VAR_NAME]
