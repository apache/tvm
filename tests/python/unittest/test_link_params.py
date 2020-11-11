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
import collections
import ctypes
import json
import os
import re
import struct
import sys

import numpy as np
import pytest

import tvm
import tvm.relay
import tvm.testing
from tvm.contrib import utils


TEST_SHAPE = (3, 4, 5)


# The data types that are linkable.
LINKABLE_DTYPES = (
    [f'uint{b}' for b in (8, 16, 32, 64)] +
    [f'int{b}' for b in (8, 16, 32, 64)] +
    ['float32', 'float64'])



def dtype_info(dtype):
    """Lookup numpy type info for the given string dtype (of LINKABLE_DTYPES above)."""
    if 'int' in dtype:
        return np.iinfo(getattr(np, dtype))
    else:
        return np.finfo(getattr(np, dtype))


# Note: for debugging, set this to an integer (i.e. 1.0). Then all "random" tensors will become
# predictable
RANDOM_TENSOR_START = None


def _make_random_tensor(dtype):
    """Create a random test tensor of shape TEST_SHAPE and the given dtype."""
    global RAND_SEED
    if RANDOM_TENSOR_START is not None:
      to_return = np.arange(RANDOM_TENSOR_START,
                            RANDOM_TENSOR_START + np.prod(TEST_SHAPE),
                            dtype=dtype).reshape(TEST_SHAPE)
      RAND_SEED += np.prod(TEST_SHAPE)
      return to_return

    dinfo = dtype_info(dtype)
    if 'int' in dtype:
        return np.random.randint(dinfo.min, dinfo.max, TEST_SHAPE, dtype=dtype)
    else:
        to_return = np.random.uniform(0, dinfo.max, TEST_SHAPE)
#        to_return = dinfo.min + (np.random.random(TEST_SHAPE) * dinfo.max)
        np.reshape(to_return, np.prod(TEST_SHAPE))[::2] *= -1
        return to_return


def _lookup_sid(graph, name):
    """Lookup the storage id of a named parameter.

    Arguments
    ---------
    graph : dict
        Parsed JSON graph.

    name : str
        Name of the tensor parameter to lookup.

    Returns
    -------
    int :
        The storage_id of the parameter.
    """
    num_outputs_seen = 0
    for i, n in enumerate(graph['nodes']):
        if n['name'] == name:
            return graph['attrs']['storage_id'][1][num_outputs_seen]
        else:
            if 'attrs' in n and 'num_outputs' in n['attrs']:
                num_outputs_seen += n['attrs']['num_outputs']
            else:
                num_outputs_seen += 1

    raise KeyError(f'no such param: {name}')


def _get_ctypes_dtype(dt):
    """Return a ctypes c_* datatype given a string data type."""
    if 'int' in dt:
        return getattr(ctypes, f'c_{dt}')
    elif dt == 'float32':
        return ctypes.c_float
    elif dt == 'float64':
        return ctypes.c_double
    else:
        assert False, f'unknown dtype: {dt}'


def _verify_linked_param(dtype, lib, mod, graph, name):
    """Directly read memory from the linked library to verify the linked parameter is correct."""
    sid = _lookup_sid(graph, name)
    # NOTE: query_imports=True because when loading a module from disk (i.e. for C backend),
    # a GraphRuntimeFactory module is created instead of the module itself.
    param_ptr = mod.get_function("_lookup_linked_param", True)(sid)
    print('verify', param_ptr)
    arr_data = (_get_ctypes_dtype(dtype) * np.prod(TEST_SHAPE)).from_address(param_ptr.value)
    gen_param = lib.params[name]
    print('gen param dtype', gen_param.dtype)
    arr = np.ndarray(
        shape=gen_param.shape, dtype=gen_param.dtype, buffer=arr_data, order='C')
    if 'int' in gen_param.dtype:
        np.testing.assert_equal(gen_param.asnumpy(), arr)
    else:
        np.testing.assert_allclose(gen_param.asnumpy(), arr)


def _make_mod_and_params(dtype):
    """Create a Relay module and parameters to test the given datatype."""
    param_decls = collections.OrderedDict()
    param_init = {}

    def _add_decl(name, dtype):
        param_decls[name] = f'%{name} : Tensor[{TEST_SHAPE}, {dtype}]'
        param_init[name] = _make_random_tensor(dtype)

    _add_decl(f'{dtype}_a', dtype)
    _add_decl(f'{dtype}_b', dtype)

    mod_lines = [
        '#[version = "0.0.5"]',
        f"def @main(%rand_input : Tensor[{TEST_SHAPE}, {dtype}], { ', '.join(param_decls.values()) } )  {{",
    ]
    if 'int' in dtype:
        mod_lines.append(
#            f'    %0 = bitwise_xor(%rand_input, bitwise_xor(%{dtype}_a, %{dtype}_b));')
            f'    %0 = add(%rand_input, %{dtype}_a);')
    else:
        mod_lines.append(
            f'    %0 = cast(add(%rand_input, cast(add(%{dtype}_a, %{dtype}_b), dtype="{dtype}")), dtype="{dtype}");')
#             f'    %0 = cast(add(%rand_input, %{dtype}_a), dtype="{dtype}");')
    mod_lines.extend([
        '    %0',
        '}'
    ])

    mod = tvm.parser.fromtext('\n'.join(mod_lines))
    return mod, param_init


@tvm.testing.requires_llvm
def test_llvm_link_params():
    for dtype in LINKABLE_DTYPES:
        mod, param_init = _make_mod_and_params(dtype)
        rand_input = _make_random_tensor(dtype)
        main_func = mod['main']
        target = 'llvm --runtime=c --system-lib --link-params'
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build(mod, target, params=param_init)
            assert set(lib.params.keys()) == {"p0"}  # NOTE: op folded

            graph = json.loads(lib.graph_json)
            for p in lib.params:
                _verify_linked_param(dtype, lib, lib.lib, graph, p)

            # Wrap in function to explicitly deallocate the runtime.
            def _run_linked(lib):
                graph_json, mod, _ = lib
                graph_rt = tvm.contrib.graph_runtime.create(graph_json, mod, tvm.cpu(0))
                graph_rt.set_input('rand_input', rand_input) # NOTE: params not required.
                graph_rt.run()
                return graph_rt.get_output(0)

            linked_output = _run_linked(lib)

        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build(mod, 'llvm --system-lib', params=param_init)

            def _run_unlinked(lib):
                graph_json, mod, lowered_params = lib
                graph_rt = tvm.contrib.graph_runtime.create(graph_json, mod, tvm.cpu(0))
                graph_rt.set_input('rand_input', rand_input, **lowered_params)
                graph_rt.run()
                return graph_rt.get_output(0)

            unlinked_output = _run_unlinked(lib)

        if 'int' in dtype:
            np.testing.assert_equal(unlinked_output.asnumpy(), linked_output.asnumpy())
        else:
            np.testing.assert_allclose(unlinked_output.asnumpy(), linked_output.asnumpy())


def _get_c_datatype(dtype):
  """Translate LINKABLE_DTYPES element to c datatype."""
  if 'int' in dtype:
    return f'{dtype}_t'
  elif dtype == 'float32':
    return 'float'
  elif dtype == 'float64':
    return 'double'
  else:
    assert False, f'unknown dtype {dtype}'


def _format_c_value(dtype, width, x):
  if 'int' in dtype:
    hex_formatstr = f'{{:{"+" if dtype.startswith("int") else ""}#0{width}x}}'
    return hex_formatstr.format(x)
  elif 'float' in dtype:
    to_ret = float(x).hex()
    if 'inf' in to_ret:
      return ('-' if x < 0 else '') + 'INFINITY'
    elif 'nan' in to_ret:
      return 'NAN'

    before, after = to_ret.split('p')
    return f'{before.rstrip("0")}p{after}'
  else:
    assert False, f"don't know dtype {dtype}"


HEX_NUM_RE = re.compile(r'[+\-]?(?:(?:0x[0-9A-Fa-f.p+-]+)|(?:INFINITY)|(?:NAN))')


def test_c_link_params():
    temp_dir = utils.tempdir()
    for dtype in LINKABLE_DTYPES:
        mod, param_init = _make_mod_and_params(dtype)
        rand_input = _make_random_tensor(dtype)
        main_func = mod['main']
        target = 'c --link-params'
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            lib = tvm.relay.build(mod, target, params=param_init)
            assert set(lib.params.keys()) == {"p0"}  # NOTE: op folded

            src = lib.lib.get_source()
            lib.lib.save('test.c', 'cc')
            c_dtype = _get_c_datatype(dtype)
            src_lines = src.split('\n')
            param = lib.params['p0'].asnumpy().reshape(np.prod(TEST_SHAPE))
            param_def = f'static const {c_dtype} __tvm_param__p0[{np.prod(param.shape)}] = {{'
            for i, line in enumerate(src_lines):
              if line == param_def:
                i += 1
                break
            else:
              assert False, f'did not find parameter definition "{param_def}":\n{src}'

            cursor = 0
            width = dtype_info(dtype).bits // 4 + 2
            if dtype.startswith("int"):
              width += 1  # Account for sign

            print('check printing of', param)
            while '};' not in src_lines[i]:
              for match in HEX_NUM_RE.finditer(src_lines[i]):
                assert match.group() == _format_c_value(dtype, width, param[cursor]), (
                  f'p0 byte {cursor}: want "{_format_c_value(dtype, width, param[cursor])}" got '
                  f'"{match.group(0)}"; full p0 follows:\n{src}')
                cursor += 1
              i += 1

            assert cursor == np.prod(param.shape)
            temp = utils.tempdir()

            # Need a unique name per library to avoid dlopen caching the lib load.
            lib_path = temp_dir.relpath(f'test-{dtype}-linked.so')
            lib['remove_params']().export_library(lib_path)
            lib_mod = tvm.runtime.load_module(lib_path)

#            lib_mod = lib_factory['default']()
            graph = json.loads(lib.graph_json)
            for p in lib.params:
                _verify_linked_param(dtype, lib, lib_mod, graph, p)

            # Wrap in function to explicitly deallocate the runtime.
            def _run_linked(lib_mod):
                graph_rt = tvm.contrib.graph_runtime.GraphModule(
                  lib_mod['default'](tvm.cpu(0)))
                graph_rt.set_input('rand_input', rand_input) # NOTE: params not required.
                print('linked', graph_rt.get_input('p0'))
                graph_rt.run()

                return graph_rt.get_output(0)

            linked_output = _run_linked(lib_mod)

        linked_params = lib.params
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            lib = tvm.relay.build(mod, 'c', params=param_init)
            _, _, params = lib
            # Need a unique name per library to avoid dlopen caching the lib load.
            lib_path = temp_dir.relpath(f'test-{dtype}-unlinked.so')
            lib.export_library(lib_path)
            lib_mod = tvm.runtime.load_module(lib_path)

            print('unlinked', params)
            def _run_unlinked(lib_mod):
                graph_rt = tvm.contrib.graph_runtime.GraphModule(lib_mod['default'](tvm.cpu(0)))
                graph_rt.set_input('rand_input', rand_input, **params)
                graph_rt.run()
                return graph_rt.get_output(0)

            unlinked_output = _run_unlinked(lib_mod)

        if 'int' in dtype:
            np.testing.assert_equal(unlinked_output.asnumpy(), linked_output.asnumpy())
        else:
            np.testing.assert_allclose(unlinked_output.asnumpy(), linked_output.asnumpy())


@tvm.testing.requires_micro
def test_crt_link_params():
    import tvm.micro


    for dtype in LINKABLE_DTYPES:
        mod, param_init = _make_mod_and_params(dtype)
        rand_input = _make_random_tensor(dtype)
        main_func = mod['main']
        target = 'c -mcpu=native --system-lib --runtime=c --link-params'
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            graph_json, lib, params = tvm.relay.build(mod, target, params=param_init)
            assert set(params.keys()) == {"p0"}  # NOTE: op folded

            workspace = tvm.micro.Workspace()
            compiler = tvm.micro.DefaultCompiler(target=target)
            opts = tvm.micro.default_options(os.path.join(tvm.micro.CRT_ROOT_DIR, "host"))
            opts['bin_opts']['ldflags'].append('-DTVM_HOST_USE_GRAPH_RUNTIME_MODULE')

            micro_binary = tvm.micro.build_static_runtime(
                # the x86 compiler *expects* you to give the exact same dictionary for both
                # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
                # the binary compiler is expecting those mutations to be in bin_opts.
                # TODO(weberlo) fix this very bizarre behavior
                workspace,
                compiler,
                lib,
                lib_opts=opts["bin_opts"],
                bin_opts=opts["bin_opts"],
                extra_libs=[os.path.join(tvm.micro.CRT_ROOT_DIR, m)
                            for m in ('graph_runtime', 'graph_runtime_module')],
            )

            flasher_kw = {
                "debug": False,
            }
            flasher = compiler.flasher(**flasher_kw)
            with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
                rpc_lib = sess.get_system_lib()
                graph_rt = tvm.contrib.graph_runtime.create(
                  graph_json, rpc_lib, sess.context)

                # NOTE: not setting params here.
                graph_rt.set_input('rand_input', rand_input)
                graph_rt.run()
                linked_output = graph_rt.get_output(0).asnumpy()

        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build(mod, 'llvm --system-lib', params=param_init)

            def _run_unlinked(lib):
                graph_json, mod, lowered_params = lib
                graph_rt = tvm.contrib.graph_runtime.create(graph_json, mod, tvm.cpu(0))
                graph_rt.set_input('rand_input', rand_input, **lowered_params)
                graph_rt.run()
                return graph_rt.get_output(0)

            unlinked_output = _run_unlinked(lib).asnumpy()

        if 'int' in dtype:
            np.testing.assert_equal(unlinked_output, linked_output)
        else:
            np.testing.assert_allclose(unlinked_output, linked_output)


if __name__ == '__main__':
  sys.exit(pytest.main(sys.argv[1:]))
