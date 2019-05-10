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
from ast import alias, Assign, keyword, Load, Name, NameConstant, Num, Return, Store, Str
import re

import astor
from tvm import relay
from tvm.relay.adt import Constructor, Pattern
from tvm.relay.expr import Expr, Function
from tvm.relay.expr_functor import ExprFunctor

INTERPRETER_VAR = '_INTRP'
OUTPUT_VAR_NAME = '_py_out'
MODULE_NAME = '_mod'

# corresponds to:
#     import tvm
#     from tvm import relay
#     from tvm.relay.backend.interpreter import RefValue, TupleValue, TensorValue, ConstructorValue
#     from tvm.relay import create_executor
#     _INTRP = create_executor(mod=_mod)
PROLOGUE = [
    ast.Import([alias('tvm', None)]),
    ast.ImportFrom('tvm', [alias('relay', None)], 0),
    ast.ImportFrom('tvm.relay.backend.interpreter',
                   [alias('RefValue', None),
                    alias('TupleValue', None),
                    alias('TensorValue', None),
                    alias('ConstructorValue', None)],
                   0),
    ast.ImportFrom('tvm.relay', [alias('create_executor', None)], 0),
    Assign([Name(INTERPRETER_VAR, Store())],
           ast.Call(Name('create_executor', Load()),
                    [], [keyword('mod',
                                 Name(MODULE_NAME, Load()))]))
]

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
        body = []
        body += PROLOGUE
        body += self.convert_module()

        prog_body, extra_defs = self.visit(check)
        body += extra_defs

        # we finally must assign the final expression to the output var
        # so it can be read after running EXEC
        body.append(Assign([Name(OUTPUT_VAR_NAME, Store())], prog_body))

        return ast.fix_missing_locations(ast.Module(body=body))


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
        for i in range(len(attributes - 1)):
            ret = ast.Attribute(ret, attributes[i+1], Load())
        return ret


    # given an attribute to a call node, generates a Python constant
    # corresponding to that attr
    def parse_attr(self, attr):
        if isinstance(attr, (int, long, float)):
            return Num(attr)
        if isinstance(attr, (list, tuple)):
            return ast.List([self.parse_attr(mem) for mem in attr],
                            Load())
        # this may fail on exotic attributes, but most cases that are
        # not the above are, in fact, strings
        return Str(repr(attr))


    # Given a Numpy array, produces an appropriate Python array
    # or numerical literal representing its contents
    def parse_numpy_array(self, arr):
        if arr.ndim == 0:
            return Num(arr.item())
        if arr.ndim == 1:
            return ast.List([Num(i) for i in arr], Load())

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


    # converts all the global functions defined in the module and returns
    # them as a list of definitions
    def convert_module(self):
        defs = []
        for var, func in self.mod.functions.items():
            converted_func, func_name = self.convert_func_node(var.name_hint, func)
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
                          None, [], None, [], []),
            body, [], None)


    def create_op_call(self, op: Expr, num_args: int, attrs):
        '''Wraps a call to an operator with an invocation of the interpreter,
        like in the tests. This is pretty dirty but is the simplest way to
        invoke operators from Python code'''

        arg_names = [self.generate_var_name('_{}_arg_{}'.format(op.name, i))
                     for i in range(num_args)]
        var_names = [self.generate_var_name('_{}_var_{}'.format(op.name, i))
                     for i in range(num_args)]
        call_name = self.generate_var_name('_{}_call'.format(op.name))

        body = [
            Assign(
                [Name(name, Store())],
                self.create_call('relay.var', [Str(name)]))
            for name in var_names
        ]

        # equiv: call = relay.op(relay_vars, attr=value)
        call_assignment = Assign(
            [Name(call_name, Store())],
            ast.Call(self.parse_name('relay.' + op.name),
                     [Name(name, Store()) for name in var_names],
                     [keyword(key, convert_attr(attrs[key])) for key in attrs.keys()]))
        body.append(call_assignment)

        # equiv: return _INTRP.evaluate(call, { relay_var : argument })
        arg_dict = ast.Dict([Name(name, Load()) for name in var_names],
                            [Name(name, Load()) for name in arg_names])
        intrp_call = self.create_call(INTERPRETER_VAR, [
            Name(call_name, Load()),
            arg_dict
        ])
        body.append(Return(intrp_call))

        func_name = self.generate_function_name('_op_call_{}'.format(op_name))
        func = self.create_def(func_name, arg_names, body)
        return (func, func_name)


    def create_constructor(self, ctor: Constructor):
        '''Given an ADT constructor, creates a Python AST node that
        obtains a reference to the same constructor'''
        type_data = self.mod[ctor.belongs_to]
        ctor_index = -1
        for i in range(len(type_data.constructors)):
            if type_data.constructors[i] == ctor:
                ctor_index = i
                break
        assert ctor_index >= 0

        # reference to type var: mod.get_global_type_var({var name})
        # reference to constructor object: mod[{type_var}].constructors[{index}]
        type_var_ref = self.create_call('{}.get_global_type_var'.format(MODULE_NAME),
                                        [Str(type_name)])
        type_data_ref = ast.Subscript(Name(MODULE_NAME, Load()),
                                      ast.Index(type_var_ref),
                                      Load())
        constructors_list_ref = ast.Attribute(type_data_ref, 'constructors', Load())
        return ast.Subscript(constructors_list_ref,
                             ast.Index(Num(ctor_index)),
                             Load())


    def create_match_check(self, pattern: Pattern):
        '''Given an ADT match pattern, this generates a Python AST
        corresponding to a function that checks if its ADT-valued
        argument matches the pattern (returning True or False).
        Returns a list of function definitions and the name of the
        function that will check whether an ADT value matches the
        given pattern'''

        func_name = self.generate_function_name('_pattern_match_check')
        arg_name = self.generate_variable_name('_pattern_match_check')

        # wildcard or var match everything
        if isinstance(pattern, (relay.PatternWildcard, relay.PatternVar)):
            return ([
                self.create_def(func_name, [arg_name],
                                [Return(NameConstant('True'))])
            ],
                    func_name)

        # constructor patterns check whether the constructors match
        # and also the matches of any nested patterns
        defs = []
        pattern_ctor = self.create_constructor(pattern.constructor)
        test = ast.Compare(Name(arg_name, Load()),
                           [ast.NotEq()], [pattern_ctor])

        comparison = ast.If(test, [Return(NameConstant(False))], [])
        body = [comparison]

        # now add checks for any nested patterns that we perform
        # after the comparison if the constructors match
        for i in range(len(pattern.patterns)):
            nested_pat = pattern.patterns[i]

            # can safely skip var or wildcard patterns: they will
            # never cause a check to fail
            if not isinstance(nested_pat, relay.PatternConstructor):
                continue

            nested_defs, nested_func = self.create_match_check(nested_pat)
            defs += nested_defs

            # equiv: if not match_func(arg.fields[i]): return False
            field_index = ast.Subscript(self.parse_name('{}.fields'.format(arg_name)),
                                        ast.Index(Num(i)),
                                        Load())
            nested_test = ast.UnaryOp(ast.Not(),
                                      self.create_call(nested_func,
                                                       [field_index]))
            nested_comparison = ast.If(nested_test, [Return(NameConstant(False))])
            body.append(nested_comparison)

        # after all checks, we return True if we have a final match
        body.append(Return(NameConstant(True)))

        final_def = self.create_def(func_name, [arg_name], body)
        defs.append(final_def)
        return (defs, func_name)


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
                return [Assign([Name('_', Store())], val)]
            if isinstance(pat, relay.PatternVar):
                return [Assign([self.include_var(pat.var, assign=True)], val)]
            # constructor pattern: assign each field of the value
            # based on subpatterns
            assignments = []
            for i in range(len(pat.patterns)):
                # we want the assignments for val.fields[i]
                field = ast.Subscript(ast.Attribute(val, 'fields', Load()),
                                      ast.Index(Num(i)), ctx.Load())
                assignments += collect_var_assignments(pat, field)
            return assignments

        func_name = self.generate_func_name('_match_clause_body')
        arg_name = self.generate_var_name('_match_clause_body')

        clause_body, defs = self.visit(body)
        assignments = collect_var_assignments(pattern, Name(arg_name, Load()))

        func_def = self.create_def(func_name, [arg_name],
                                   defs + assignments + [Return(clause_body)])
        return ([func_def], func_name)


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
        let var = body in value
        =======================
        def let_thunk(var):
            return value
        let_thunk(body)
        '''
        value_body, value_defs = self.visit(letexp.value)

        func_name = self.generate_function_name('_let_func')
        binding_func = self.create_def(func_name, [self.get_var_name(letexp.var)],
                                       value_defs + [Return(value_body)])

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
        ret = ast.Subscript(tup, ast.Index(Num(tgi.index)), Load())
        return (ret, tup_defs)


    def visit_if(self, if_block: Expr):
        cond_body, cond_defs = self.visit(if_block.cond)
        true_body, true_defs = self.visit(if_block.true_branch)
        false_body, false_defs = self.visit(if_block.false_branch)
        ret = ast.IfExp(cond_body, true_body, false_body)
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
        return (self.include_var(func_name), [converted_func])


    def visit_call(self, call: Expr):
        '''For calls, we must distinguish between ordinary functions,
        operators, and constructor calls.'''
        func = call.op
        fields, field_defs = self.convert_fields(call.args)
        if isinstance(func, relay.Op):
            op_func, op_call = self.create_op_call(func, len(call.args), call.attrs)
            defs = [op_func] + field_defs
            return (self.create_call(op_call, fields), defs)

        if isinstance(func, relay.Constructor):
            # produce a constructor value
            return (self.create_call('ConstructorValue',
                                     [self.create_constructor(func),
                                      ast.List(fields, Load())]),
                    field_defs)

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

        thunk_body = []
        for clause in match.clauses:
            checker_defs, check_name = self.create_match_check(clause.pattern)
            defs += checker_defs
            body_def, body_name = self.create_match_clause_body(clause.pattern, clause.body)
            defs.append(body_def)

            # equiv: if check(data): return body(data)
            thunk_body.append(ast.If(
                self.generate_call(check_name, [data]),
                [Return(self.generate_call(body_name, [data]))]
            ))

        # finally if nothing matches we have a failed assert
        # (should never happen)
        thunk_body.append(ast.Assert(NameConstant(False)),
                          Str('Match was not exhaustive'))

        thunk_name = self.generate_func_name('_match_thunk')
        thunk_def = self.create_def(thunk_name, [], defs + thunk_body)
        return (self.create_call(thunk_name, []), [thunk_def])

    # these are both handled in the "call" case
    def visit_constructor(self, _):
        pass
    def visit_op(self, _):
        pass


def to_python(expr: Expr, mod=relay.Module()) -> str:
    '''Converts the given Relay expression into a Python script.'''
    converter = PythonConverter(mod)
    py_ast = converter.convert(expr)
    return astor.to_source(py_ast)


def run_as_python(expr: Expr, mod=relay.Module()):
    '''Converts the given Relay expression into a Python script and
    executes it.'''
    py_ast = to_python(expr, mod)
    code = compile(py_ast, '<string>', 'exec')
    var_map = {OUTPUT_VAR_NAME : None, MODULE_NAME : mod}
    #pylint: disable=exec-used
    exec(code, {}, var_map)
    return output_map[OUTPUT_VAR_NAME]
