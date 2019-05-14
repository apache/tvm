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
from ast import alias, Assign, Load, Name, NameConstant, Num, Return, Store, Str
import re

import numpy
import tvm
from tvm import relay
from tvm.relay.adt import Constructor, Pattern
from tvm.relay.backend import compile_engine
from tvm.relay.expr import Expr, Function
from tvm.relay.expr_functor import ExprFunctor

OUTPUT_VAR_NAME = '_py_out'
MODULE_NAME = '_mod'

# corresponds to:
#     import numpy
#     import tvm
#     from tvm import relay
#     from tvm.relay.backend.interpreter import RefValue, TupleValue, TensorValue, ConstructorValue
PROLOGUE = [
    ast.Import([alias('numpy', None)]),
    ast.Import([alias('tvm', None)]),
    ast.ImportFrom('tvm', [alias('relay', None)], 0),
    ast.ImportFrom('tvm.relay.backend.interpreter',
                   [alias('RefValue', None),
                    alias('TupleValue', None),
                    alias('TensorValue', None),
                    alias('ConstructorValue', None)],
                   0)
]

class PythonConverter(ExprFunctor):
    '''Functor for translating Relay programs into Python ASTs.'''

    def __init__(self, mod, target) -> None:
        super().__init__()
        self.mod = mod
        self.tgt = target
        self.engine = compile_engine.get()
        self.fun_no = 0
        self.var_no = 0
        self.var_map = {}


    def convert(self, prog: Expr):
        '''This method converts the passed Relay expression into a Python
        AST object with equivalent semantics.

        The Python AST can be executed using exec(); it can be turned
        into text and inspected using astor.
        '''
        optimized = self.optimize(prog)

        # start with conversion prelude (imports) and convert global defs
        body = []
        body += PROLOGUE
        body += self.convert_module()

        prog_body, extra_defs = self.visit(optimized)
        body += extra_defs

        # we finally must assign the final expression to the output var
        # so it can be read after running EXEC
        body.append(Assign([Name(OUTPUT_VAR_NAME, Store())], prog_body))

        return ast.fix_missing_locations(ast.Module(body=body))


    def optimize(self, prog: Expr):
        '''Performs optimizations necessary to be able to generate code for prog.'''
        check = relay.ir_pass.infer_type(prog, self.mod)
        assert relay.ir_pass.well_formed(check)
        # necessary pass: SimplifyInference (otherwise we can't generate code for some operators)
        # and fusion (to get primitive functions)
        simplify = relay.ir_pass.simplify_inference(check)
        simplify_checked = relay.ir_pass.infer_type(simplify, self.mod)
        fused = relay.ir_pass.fuse_ops(simplify_checked)
        fused_checked = relay.ir_pass.infer_type(fused, self.mod)
        assert relay.ir_pass.well_formed(fused_checked)
        return fused_checked


    # Only checks that the name does not contain invalid characters
    # (underscores, numbers, and letters are permitted). Since
    # we append a number and underscore to var names anyway, it doesn't
    # matter if the name is the empty string
    def check_safe_name(self, name: str):
        return re.match(r'\w*$', name)


    # generates a unique variable name starting from the hint
    def generate_var_name(self, name_hint: str):
        if not self.check_safe_name(name_hint):
            raise Exception('Name hint contains invalid characters: {}'.format(name_hint))
        name = '{}_var_{}'.format(name_hint, self.var_no)
        self.var_no += 1
        return name


    # generates a unique function name starting from the hint
    def generate_function_name(self, name_hint: str):
        if not self.check_safe_name(name_hint):
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


    # returns n variable AST node for the given Relay var depending on
    # whether it must appear in an assignment or not
    def include_var(self, var: Expr, assign=False):
        name = self.get_var_name(var)
        return Name(name, Store() if assign else Load())


    # Given the name of a Python method with dots (e.g.,
    # 'relay.var'), returns an appropriate AST object corresponding
    # to that name
    def parse_name(self, name: str):
        attributes = name.split('.')
        ret = Name(attributes[0], Load())
        for i in range(len(attributes) - 1):
            ret = ast.Attribute(ret, attributes[i+1], Load())
        return ret


    # given an attribute to a call node, generates a Python constant
    # corresponding to that attr
    def parse_attr(self, attr):
        if isinstance(attr, bool):
            return NameConstant(attr)
        if isinstance(attr, (int, long, float)):
            return Num(attr)
        if isinstance(attr, (list, tuple)):
            return ast.List([self.parse_attr(mem) for mem in attr], Load())
        # this may fail on exotic attributes, but most cases that are
        # not the above are, in fact, strings
        return Str(attr)


    def parse_numpy_array(self, arr):
        '''Given a Numpy array, produces an appropriate Python array
        or numerical literal representing its contents'''
        # cannot use Num for bools
        parse_single = lambda i: NameConstant(i) if isinstance(i, bool) else Num(i)
        if arr.ndim == 0:
            return parse_single(arr.item())
        if arr.ndim == 1:
            return ast.List([parse_single(i) for i in arr], Load())

        elts = []
        for row in arr:
            elts.append(numpy_to_array(row))
        return ast.List(elts, Load())


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
        thunk = self.create_def(thunk_name, [], defs + [Return(body)])
        return (thunk, thunk_name)


    # converts the given Relay function into a Python function
    def convert_func_node(self, name_hint: str, func: Function):
        func_name = self.generate_function_name(name_hint)
        var_names = [self.get_var_name(var) for var in func.params]
        body, defs = self.visit(func.body)
        ret = self.create_def(func_name, var_names, defs + [Return(body)])
        return (ret, func_name)


    def convert_module(self):
        '''Converts all the global functions defined in the module and returns
        them as a list of definitions'''
        defs = []
        for var, func in self.mod.functions.items():
            # optimize the definition so any operators used are lowered
            opt_func = self.optimize(func)
            converted_func, func_name = self.convert_func_node(var.name_hint, opt_func)
            defs.append(converted_func)
            # need to add this assignment so references to the global var in the program
            # go to the function!
            defs.append(Assign([Name(var.name_hint, Store())],
                               Name(func_name, Load())))
        return defs


    # simple function call
    def create_call(self, func_name: str, arguments):
        return ast.Call(self.parse_name(func_name), arguments, [])


    # wrapper over raw AST node, whose constructor is inconvenient
    def create_def(self, func_name: str, arguments: [str], body):
        return ast.FunctionDef(
            func_name,
            ast.arguments([ast.arg(argument, None)
                           for argument in arguments],
                          None, [], [], None, []),
            body, [], None)


    def create_op_call(self, op: Function, relay_args, py_args):
        '''Lowers the passed primitive function, registers it in TVM's
        global compiler, and produces a call to the lowered function in
        the generated Python code.'''

        # compile the function and register globally
        cc_key = compile_engine.CCacheKey(func, self.tgt)
        func_hash = relay.ir_pass.structural_hash(func)
        op_name = '_lowered_op_{}'.format(func_hash)
        if not tvm.get_global_func(op_name, allow_missing=True):
            jitted = self.engine.jit(cc_key, self.tgt)
            tvm.register_func(op_name, jitted)

        # use the types of the function arguments to determine whether we expect
        # a tensor or tuple (returns list of inputs to the lowered op call)
        def convert_input(py_input, arg_type):
            # equivalent: input.data
            if isinstance(arg_type, relay.TensorType):
                return [Attribute(py_input, 'data', Load())]
            assert isinstance(arg_type, relay.TupleType)
            # convert each input.fields[i]
            return [
                convert_input(
                    ast.Subscript(ast.Attribute(py_input, 'fields', Load()),
                                  ast.Index(Num(i)), Load()),
                    type.fields[i])
                for i in range(len(input_fields))
            ]

        # use the function return type to produce auxiliary variables to store outputs
        # returns: ([assignments of output vars],
        #           [extra arguments to pass to op call],
        #           expression collecting output)
        def convert_output(ret_type):
            if isinstance(ret_type, relay.TensorType):
                output_var_name = self.generate_var_name('_{}_out_var'.format(op_name))
                output_var = Name(output_var_name, Load())
                shape = ast.Tuple([dim for dim in type.concrete_shape()], Load())
                # create a new ND array of the right shape and dtype
                assign_output = Assign(
                    [Name(output_var_name, Store())],
                    self.create_call('numpy.empty', [shape, Str(type.dtype)]))
                # we pass the data field as an argument
                extra_arg = ast.Attribute(output_var, 'data', Load())
                return ([assign_output], [extra_arg], output_var)
            assert isinstance(ret_type, relay.TupleType)
            assignments = []
            extra_args = []
            fields = []
            for t in type.fields:
                inner_assignments, inner_args, inner_output = convert_output(t)
                assignments += inner_assignments
                extra_args += inner_args
                fields.append(inner_output)
            return (assignments, extra_args, self.create_call('TupleValue', fields))

        # create a function to wrap the call of the lowered op and return
        # a call to that function
        wrap_name = self.generate_function_name('_{}_wrapper'.format(op_name))
        wrap_args = [self.generate_var_name('_{}_wrapper_arg_i'.format(op_name)) for
                     i in range(len(py_args))]

        inner_call_args = [convert_input(Name(wrap_args[i], Load()),
                                         relay_args[i].checked_type)
                           for i in range(len(py_args))]
        output_assignments, aux_args, output = convert_output(op.checked_type.ret_type)
        # equiv: tvm.get_global_func(op_name)
        op_call = self.create_call('tvm.get_global_func', [Str(op_name)])
        inner_call = ast.Call(op_call, inner_call_args + aux_args, [])
        body = output_assignments + [ast.Expression(inner_call), Return(output)]
        wrap_def = self.create_def(wrap_name, wrap_args, body)
        return wrap_def, self.create_call(wrap_name, py_args)


    def create_constructor(self, ctor: Constructor):
        '''Given an ADT constructor, creates a Python AST node that
        obtains a reference to the same constructor'''
        type_data = self.mod[ctor.belong_to]
        ctor_index = -1
        for i in range(len(type_data.constructors)):
            if type_data.constructors[i] == ctor:
                ctor_index = i
                break
        assert ctor_index >= 0

        # reference to type var: mod.get_global_type_var({var name})
        # reference to constructor object: mod[{type_var}].constructors[{index}]
        type_name = ctor.belong_to.var.name
        type_var_ref = self.create_call('{}.get_global_type_var'.format(MODULE_NAME),
                                        [Str(type_name)])
        type_data_ref = ast.Subscript(Name(MODULE_NAME, Load()),
                                      ast.Index(type_var_ref),
                                      Load())
        constructors_list_ref = ast.Attribute(type_data_ref, 'constructors', Load())
        return ast.Subscript(constructors_list_ref,
                             ast.Index(Num(ctor_index)),
                             Load())


    def create_match_check(self, pattern: Pattern, data):
        '''Given an ADT match pattern and a (Python) expression pointing to
        an ADT value, this generates a Python expression that checks if the
        ADT value matches the given pattern (returning True or False).'''

        # wildcard or var match everything
        if isinstance(pattern, (relay.PatternWildcard, relay.PatternVar)):
            return NameConstant(True)

        # constructor patterns check whether the constructors match
        # and also the matches of any nested patterns

        # equiv: (arg.constructor == patern_constructor)
        conds = [ast.Compare(ast.Attribute(data, 'constructor', Load()),
                             [ast.Eq()], [self.create_constructor(pattern.constructor)])]

        # now check for any nested patterns
        for i in range(len(pattern.patterns)):
            nested_pat = pattern.patterns[i]
            # can safely skip var or wildcard patterns: they will
            # never cause a check to fail
            if not isinstance(nested_pat, relay.PatternConstructor):
                continue

            # index into the value corresponding to the subpattern
            field_index = ast.Subscript(ast.Attribute(data, 'fields', Load()),
                                        ast.Index(Num(i)), Load())
            conds.append(self.create_match_check(nested_pat, field_index))

        # if we do not need to check nested pattern, just return the single check
        if len(conds) == 1:
            return conds[0]
        # otherwise AND together any nested checks
        return ast.BoolOp(ast.And(), conds)


    def create_match_clause_body(self, pattern: Pattern, body: Expr):
        '''Given a match clause pattern and a clause body,
        generates a Python function that when called with an ADT
        that matches the pattern, returns the result of evaluating
        the clause body. This function returns a function definition
        and the name of the generated function.'''

        # this helper function ensures that the pattern is used to
        # properly assign all subfields of the given AST for use
        # in the clause body
        #
        # E.g., for PatternConstructor(A, PatternVar(v), PatternWildcard(),
        #   PatternConstructor(B, PatternVar(w)))
        # we would want to have
        # v = a.fields[0]
        # _ = a.fields[1]
        # w = a.fields[2].fields[0]
        def collect_var_assignments(pat, val):
            if isinstance(pat, relay.PatternWildcard):
                return []
            if isinstance(pat, relay.PatternVar):
                return [Assign([self.include_var(pat.var, assign=True)], val)]
            # constructor pattern: assign each field of the value
            # based on subpatterns
            assignments = []
            for i in range(len(pat.patterns)):
                # we want the assignments for val.fields[i]
                field = ast.Subscript(ast.Attribute(val, 'fields', Load()),
                                      ast.Index(Num(i)), Load())
                assignments += collect_var_assignments(pat.patterns[i], field)
            return assignments

        func_name = self.generate_function_name('_match_clause_body')
        arg_name = self.generate_var_name('_match_clause_body')

        clause_body, defs = self.visit(body)
        assignments = collect_var_assignments(pattern, Name(arg_name, Load()))

        func_def = self.create_def(func_name, [arg_name],
                                   defs + assignments + [Return(clause_body)])
        return (func_def, func_name)


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
        return (Name(gvar.name_hint, Load()), [])


    def visit_let(self, letexp: Expr):
        # To properly account for scoping and ensure that the entire node produces an expression,
        # we translate the let binding as a function that we call with the value we intend to bind.
        # Yes, this is somewhat ugly.
        '''
        let var = value in body
        =======================
        def let_thunk(var):
            return body
        let_thunk(value)
        '''
        bind_body, bind_defs = self.visit(letexp.body)

        func_name = self.generate_function_name('_let_func')
        binding_func = self.create_def(func_name, [self.get_var_name(letexp.var)],
                                       bind_defs + [Return(bind_body)])

        # we call the binding func with the intended value for the bound variable
        value_body, value_defs = self.visit(letexp.value)
        value_defs.append(binding_func)
        binding_call = self.create_call(func_name, [value_body])

        return (binding_call, value_defs)


    def visit_tuple(self, tup: Expr):
        fields, ret_defs = self.convert_fields(tup.fields)
        return (self.create_call('TupleValue', fields), ret_defs)


    def visit_tuple_getitem(self, tgi: Expr):
        tup, tup_defs = self.visit(tgi.tuple_value)
        ret = ast.Subscript(tup, ast.Index(Num(tgi.index)), Load())
        return (ret, tup_defs)


    def visit_if(self, if_block: Expr):
        cond_body, cond_defs = self.visit(if_block.cond)
        true_body, true_defs = self.visit(if_block.true_branch)
        false_body, false_defs = self.visit(if_block.false_branch)

        # need to get the value out of a TensorValue to check the condition
        # equvialent to: val.asnumpy()
        cond_check = ast.Call(ast.Attribute(cond_body, 'asnumpy', Load()), [], [])
        ret = ast.IfExp(cond_check, true_body, false_body)
        return (ret, cond_defs + true_defs + false_defs)


    def visit_constant(self, constant: Expr):
        '''Proceeds by converting constant value to a numpy array
        and converting it to the appropriate value in the generated
        code (whether it be a Python scalar or a Numpy array)'''

        value = constant.data.asnumpy()
        const_expr = self.create_call('numpy.array',
                                      [self.parse_numpy_array(value)])
        return (self.create_call('TensorValue', [const_expr]), [])


    def visit_function(self, func: Expr):
        # Python's lambdas are very restrictive, so we do "name" inline functions
        converted_func, func_name = self.convert_func_node('_anon_func', func)
        return (Name(func_name, Load()), [converted_func])


    def visit_call(self, call: Expr):
        '''For calls, we must distinguish between ordinary functions,
        operators, and constructor calls.'''
        func = call.op
        fields, field_defs = self.convert_fields(call.args)

        if isinstance(func, relay.Op):
            raise Exception('Operators should have been lowered and eliminated')

        if isinstance(func, relay.Constructor):
            # produce a constructor value
            return (self.create_call('ConstructorValue',
                                     [self.create_constructor(func),
                                      ast.List(fields, Load())]),
                    field_defs)

        # lowered operator: generate a call to a function that gets the PackedFunc
        # from TVM's registry
        if isinstance(func, Function) and func.attrs and func.attrs.Primitive.value == 1:
            op_call_def, op_call = self.create_op_call(func, call.args, fields)
            return (op_call, field_defs + [op_call_def])

        # ordinary function
        converted_func, defs = self.visit(func)
        defs += field_defs
        return (ast.Call(converted_func, fields, []), defs)


    def visit_ref_create(self, ref: Expr):
        val, defs = self.visit(ref.value)
        return (self.create_call('RefValue', [val]), defs)


    def visit_ref_read(self, read: Expr):
        ref, defs = self.visit(read.ref)
        return (ast.Attribute(ref, 'value', Load()), defs)


    def visit_ref_write(self, write: Expr):
        '''For writing refs, we wrap the update in a thunk
        (returning an empty tuple to match Relay's semantics)
        that we execute at the right time. This ensures such assignments
        can be properly nested, since assignments are statements
        in Python but expressions in Relay'''
        ref, ref_defs = self.visit(write.ref)
        val, val_defs = self.visit(write.value)
        thunk_name = self.generate_function_name('_ref_write_thunk')
        thunk = self.create_def(
            thunk_name, [],
            ref_defs + val_defs + [
                Assign([ast.Attribute(ref, 'value', Store())], val),
                Return(self.create_call('TupleValue', []))
            ])
        return (self.create_call(thunk_name, []), [thunk])


    def visit_match(self, match: Expr):
        '''For matches, we wrap the entire expression in a thunk
        because it is easiest to implement them using if statements.
        For each clause, we generate a function that checks if the
        pattern matches. If yes, we call a function that assigns
        the variables appropriately and invokes the clause body.'''
        data, defs = self.visit(match.data)
        data_var = self.generate_var_name('_match_data')

        # must ensure the data clause is executed exactly once
        thunk_body = [Assign([Name(data_var, Store())], data)]
        for clause in match.clauses:
            check_expr = self.create_match_check(clause.lhs, Name(data_var, Load()))
            body_def, body_name = self.create_match_clause_body(clause.lhs, clause.rhs)
            defs.append(body_def)

            # equiv: if check(data): return body(data)
            thunk_body.append(ast.If(
                check_expr,
                [Return(self.create_call(body_name, [Name(data_var, Load())]))],
                []
            ))

        # finally if nothing matches we have a failed assert (should never happen)
        thunk_body.append(ast.Assert(NameConstant(False), Str('Match was not exhaustive')))

        thunk_name = self.generate_function_name('_match_thunk')
        thunk_def = self.create_def(thunk_name, [], defs + thunk_body)
        return (self.create_call(thunk_name, []), [thunk_def])


    # these are both handled in the "call" case
    def visit_constructor(self, _):
        pass
    def visit_op(self, _):
        pass


def to_python(expr: Expr, mod=relay.Module(), target='llvm'):
    '''Converts the given Relay expression into a Python script (as a Python AST object).
    For easiest debugging, import the astor package and use to_source().'''
    converter = PythonConverter(mod, target)
    return converter.convert(expr)


def run_as_python(expr: Expr, mod=relay.Module(), target='llvm'):
    '''Converts the given Relay expression into a Python script and
    executes it.'''
    py_ast = to_python(expr, mod, target)
    code = compile(py_ast, '<string>', 'exec')
    # must pass in imports in globals dict or else nested functions
    # won't be able to call them unless they are explicitly made globals
    # (weird quirk of Python ASTs)
    imports = {
        'numpy': numpy,
        'tvm': tvm,
        'relay': relay,
        'TensorValue': relay.backend.interpreter.TensorValue,
        'TupleValue': relay.backend.interpreter.TupleValue,
        'ConstructorValue': relay.backend.interpreter.ConstructorValue,
        'RefValue': relay.backend.interpreter.RefValue,
        MODULE_NAME : mod
    }
    var_map = {OUTPUT_VAR_NAME : None}
    #pylint: disable=exec-used
    exec(code, imports, var_map)
    return var_map[OUTPUT_VAR_NAME]
