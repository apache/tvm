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
"""
Defines the entry point into the AoT compiler.
"""
import ctypes
import os
import subprocess
import tempfile
import time

import tvm
from tvm import relay, get_global_func, register_func
from tvm.relay.function import Function
from tvm.relay.expr import Expr, Let, GlobalVar
from tvm.relay.adt import Constructor
from tvm.relay.expr_functor import ExprFunctor
from tvm.relay.backend import compile_engine
from .little_cpp import (PackedCall, CPPFunction, Invoke, Decl, CPPIf,
                         CPPTuple, CPPMatch, CPPConstructor, CPPTupleGetItem,
                         CPPRefCreate, CPPRefRead, CPPRefWrite)
from . import to_source
from .convert import convert

TVM_PATH = os.environ['TVM_HOME']

def must_run_process(args):
    proc = subprocess.run(args, check=True)
    assert proc.returncode == 0

def compile_cpp(source, lib_name, flags=None, lib_path=None):
    """
    Compiles the given source into a C++ library
    and returns the full path to the compiled library.
    """
    if flags is None:
        flags = []

    if lib_path is None:
        lib_path = os.curdir

    debug_source_path = os.path.join(lib_path, 'source.cc')
    # Write out the file for debugging.
    with open(debug_source_path, 'w') as source_file:
        source_file.write(source)

    # with tempfile.TmporaryDirectory() as tmpdir:
    tmpdir = tempfile.mkdtemp(prefix="relay_aot_compiler")
    lib_path = os.path.join(tmpdir, lib_name)
    source_path = os.path.join(tmpdir, 'source.cc')
    with open(source_path, 'w') as source_file:
        source_file.write(source)

    must_run_process(["clang-format", "-i", debug_source_path])

    system = os.uname()[0]
    include_paths = [
        f"-I{TVM_PATH}/3rdparty/dmlc-core/include",
        f"-I{TVM_PATH}/3rdparty/dlpack/include",
        f"-I{TVM_PATH}/3rdparty/HalideIR/src",
        f"-I{TVM_PATH}/include",
        f"-L{TVM_PATH}/build"
    ]

    if system == 'Darwin':
        command = [
            "clang", "-std=c++14", "-shared", "-undefined", "dynamic_lookup",
            "-o", lib_path,
            source_path,
            *include_paths,
            "-ltvm"
        ] + flags
    else:
        command = [
            "clang", "-std=c++14", "-shared", "-fPIC", "-o", lib_path,
            source_path,
            *include_paths,
            "-ltvm"
        ] + flags

    must_run_process(command)
    return lib_path

def load_lib(name):
    return ctypes.CDLL(name, ctypes.RTLD_GLOBAL)

def is_primitive(expr: relay.Expr):
    return (isinstance(expr, relay.Function)
            and expr.attrs
            and expr.attrs.Primitive.value == 1)

class AoTCompiler(ExprFunctor):
    """
    Takes a Relay program and converts into a Little CPP program
    that can in turn be converted into C++ source code.
    """
    def __init__(self, mod, tgt) -> None:
        super().__init__()
        self.mod = mod
        self.tgt = tgt
        self.engine = compile_engine.get()
        self.bindings = [[]]
        self.gv_map = {}

    def add_binding(self, var, value):
        self.bindings[-1].append((var, value))

    def optimize(self, expr: Function) -> Function:
        opts = tvm.transform.Sequential([
            relay.transform.SimplifyInference(),
            relay.transform.FuseOps(),
            relay.transform.ToANormalForm()])
        self.mod['main'] = expr
        self.mod = opts(self.mod)
        ret = self.mod['main']
        return ret

    def mk_primitive_op(self, func: Expr, args, output_type) -> Expr:
        cc_key = compile_engine.CCacheKey(func, self.tgt)
        func_hash = tvm.ir.structural_hash(func)
        name = f"op_{func_hash}"
        if not get_global_func(name, allow_missing=True):
            jit_func = self.engine.jit(cc_key, self.tgt)
            register_func(name, jit_func)
        return PackedCall(name, args, [x.checked_type for x in args], output_type)

    def visit_call(self, call: Expr) -> Expr:
        if is_primitive(call.op):
            return self.mk_primitive_op(call.op, call.args, call.checked_type)
        if isinstance(call.op, Constructor):
            return CPPConstructor(call.op.tag, [self.visit(arg) for arg in call.args])
        assert call.attrs is None
        args = [self.visit(arg) for arg in call.args]
        func = self.visit(call.op)
        return Invoke(func, args)

    def visit_let(self, let: Expr) -> Expr:
        self.bindings.append([])

        while isinstance(let, Let):
            cpp_value = self.visit(let.value)
            self.add_binding(let.var, cpp_value)
            let = let.body

        bindings = self.bindings.pop()
        body = self.visit(let)

        return Decl(bindings, body)

    def visit_var(self, var):
        return var

    def visit_global_var(self, gv):
        if gv not in self.gv_map:
            self.gv_map[gv] = "to be updated"
            self.gv_map[gv] = self.visit(self.mod[gv])
        return gv

    def visit_function(self, func):
        if is_primitive(func):
            body = self.mk_primitive_op(func, func.params, func.ret_type)
            return CPPFunction(func.params, body, func.checked_type.ret_type)
        return CPPFunction(func.params, self.visit(func.body), func.checked_type.ret_type)

    def visit_constant(self, const):
        return const

    def visit_if(self, i):
        return CPPIf(self.visit(i.cond),
                     self.visit(i.true_branch),
                     self.visit(i.false_branch),
                     i.checked_type)

    def visit_tuple(self, t):
        return CPPTuple([self.visit(f) for f in t.fields], t.checked_type)

    def visit_match(self, m):
        return CPPMatch(self.visit(m.data),
                        [(c.lhs, self.visit(c.rhs)) for c in m.clauses],
                        m.checked_type)

    def visit_op(self, op):
        raise Exception(f'op outside of primitive: {op}')

    def visit_constructor(self, ctor):
        raise Exception('Constructors should be handled when visiting calls.')

    def visit_tuple_getitem(self, t):
        return CPPTupleGetItem(self.visit(t.tuple_value), t.index, t.checked_type)

    def visit_ref_create(self, r):
        return CPPRefCreate(self.visit(r.value), r.checked_type)

    def visit_ref_read(self, r):
        return CPPRefRead(self.visit(r.ref), r.checked_type)

    def visit_ref_write(self, r):
        return CPPRefWrite(self.visit(r.ref), self.visit(r.value))

_LIB_COUNTER = 1
_LIB = []

def lib_and_func_name(name):
    global _LIB_COUNTER
    packed_name = f'relay.aot.{name}.{_LIB_COUNTER}'
    lib_name = f"librelay_aot_{_LIB_COUNTER}.so"
    _LIB_COUNTER += 1
    return lib_name, packed_name

def _mk_wrapper(func, ctx, constants, record_time):
    def _wrapper(*args):
        new_constants = [convert(a, ctx) for a in constants]
        new_args = [convert(a, ctx) for a in args]
        begin = time.perf_counter()
        res = func(*new_constants, *new_args)
        end = time.perf_counter()
        return res if not record_time else (res, end - begin)
    return _wrapper

def compile_prog(func, mod, ctx, tgt, name='default', record_time=False):
    """Compile a Relay function into a C++ file that
    implements a program with the same semantics,
    which calls into TVM only for operators.

    Parameters
    ----------
    func: Expr
        A Relay function to compile
        (either a literal Relay function
        or a GlobalVar that is in `mod`).

    mod: IRModule
        Module containing any functions referenced by `func`.

    ctx: Context
        The TVM context.

    tgt: Target
        The TVM target.

    name: String
        The name of the target binary library.

    record_time: Bool
        If True, the return value of the function
        will include the program's execution time.

    Returns
    -------
    result: Function
        A function that, when pass in some values,
        will convert them to the right format
        and call the compiled func (a PackedFunc).
    """
    global _LIB
    if isinstance(func, GlobalVar):
        func = mod[func]
    assert isinstance(func, Function)
    compiler = AoTCompiler(mod, tgt)
    func = compiler.optimize(func)
    func = compiler.visit(func)
    lib_name, packed_name = lib_and_func_name(name)
    constants, source_code = to_source.to_source(func, compiler.gv_map, ctx, packed_name)
    lib_name = f"librelay_aot_{_LIB_COUNTER}.so"
    library_path = compile_cpp(source_code, lib_name, flags=["-O3"])
    _LIB.append(load_lib(library_path))
    func = get_global_func(packed_name)
    return _mk_wrapper(func, ctx, constants, record_time)
