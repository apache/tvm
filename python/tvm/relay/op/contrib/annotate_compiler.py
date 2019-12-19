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
from __future__ import absolute_import

import logging
import pkgutil
from pathlib import Path
from importlib import import_module

from .. import op as reg

logger = logging.getLogger('AnnotateCompiler')

# Load available contrib compilers
compilers = {}
for _, name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    compilers[name] = import_module(
        '.%s' % name, package='.'.join(__name__.split('.')[:-1]))


def get_annotate_compiler(compiler, op_name):
    """Get the annotate_compiler function from the registered compilers.

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
        if hasattr(compilers[compiler], 'annotate_compiler'):
            annotate_compiler = getattr(compilers[compiler], 'annotate_compiler')
            if hasattr(annotate_compiler, op_name):
                return getattr(annotate_compiler, op_name)

    logger.warning("%s in %s is not registered. Fallback to CPU", op_name,
                   compiler)
    return lambda x, y: False


@reg.register_annotate_compiler("nn.conv2d")
def annotate_conv2d(attrs, args, compiler):
    """Check if the provided compiler should be used for conv2d.
    """
    return get_annotate_compiler(compiler, 'conv2d')(attrs, args)


@reg.register_annotate_compiler("nn.dense")
def annotate_dense(attrs, args, compiler):
    """Check if the provided compiler should be used for dense.
    """
    return get_annotate_compiler(compiler, 'dense')(attrs, args)


@reg.register_annotate_compiler("nn.relu")
def annotate_relu(attrs, args, compiler):
    """Check if the provided compiler should be used for relu.
    """
    return get_annotate_compiler(compiler, 'relu')(attrs, args)


@reg.register_annotate_compiler("nn.batch_norm")
def annotate_batch_norm(attrs, args, compiler):
    """Check if the provided compiler should be used for batch_norm.
    """
    return get_annotate_compiler(compiler, 'batch_norm')(attrs, args)


@reg.register_annotate_compiler("subtract")
def annotate_subtract(attrs, args, compiler):
    """Check if the provided compiler should be used for subtract.
    """
    return get_annotate_compiler(compiler, 'subtract')(attrs, args)


@reg.register_annotate_compiler("add")
def annotate_add(attrs, args, compiler):
    """Check if the provided compiler should be used for add.
    """
    return get_annotate_compiler(compiler, 'add')(attrs, args)


@reg.register_annotate_compiler("multiply")
def annotate_multiply(attrs, args, compiler):
    """Check if the provided compiler should be used for multiply.
    """
    return get_annotate_compiler(compiler, 'multiply')(attrs, args)
