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
# pylint: disable=invalid-name, unused-argument
"""
External compiler related feature registration.
It implements dispatchers that check if an operator should use a given compiler
to generate code.

Each compiler can customize the support of an operator. For example, they can
check the attribute of the operator and/or the features of the input arguments
to decide if we should use the compiler for codegen.
"""
import logging
import pkgutil
from pathlib import Path
from importlib import import_module

from .. import op as reg

logger = logging.getLogger('ExternalCompiler')

# Load available contrib compilers
compilers = {}
for _, name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    compilers[name] = import_module(
        '.%s' % name, package='.'.join(__name__.split('.')[:-1]))


def get_external_compiler(compiler, op_name):
    """Get the external_compiler function from the registered compilers.

    Parameters
    ----------
    compiler : Str
        The name of a compiler that is used to generate code.

    op_name : Str
        The name of an operator.

    Returns
    -------
    ret : bool
        If the operator uses the provided compiler for codegen.
    """
    if compiler in compilers:
        if hasattr(compilers[compiler], 'external_compiler'):
            external_compiler = getattr(
                compilers[compiler], 'external_compiler')
            op_name_sfx = op_name[op_name.rfind(".")+1:]
            if hasattr(external_compiler, op_name_sfx):
                return getattr(external_compiler, op_name_sfx)

    logger.warning("%s in %s is not registered. Fallback to CPU", op_name,
                   compiler)
    return lambda x, y: False


def _register_external_compiler_helper(op_name):
    """Helper function to register an operator for external compilers.

    Parameters
    ----------
    op_name : Str
        The name of an operator.

    Returns
    -------
    ret : callable
        A callable to register the operator.
    """
    @reg.register_external_compiler(op_name)
    def _register_wrapper(attrs, args, compiler):
        return get_external_compiler(compiler, op_name)(attrs, args)
    return _register_wrapper


_register_external_compiler_helper("nn.conv2d")
_register_external_compiler_helper("nn.dense")
_register_external_compiler_helper("nn.relu")
_register_external_compiler_helper("nn.batch_norm")
_register_external_compiler_helper("subtract")
_register_external_compiler_helper("add")
_register_external_compiler_helper("multiply")
