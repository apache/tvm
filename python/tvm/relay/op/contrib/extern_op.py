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

It implements dispatchers that check if an operator should use the external
codegen tool.

Each compiler can customize the support of the operator. For example, they can
check the attribute of an operator and/or the features of the input arguments
to decide if we should use the external compiler.
"""
from __future__ import absolute_import

import logging
import pkgutil
from pathlib import Path
from importlib import import_module

from .. import op as reg

logger = logging.getLogger('ExternOp')

# Load available contrib compilers
compilers = {}
for _, name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    compilers[name] = import_module('.%s' % name, package='.'.join(__name__.split('.')[:-1]))

def get_extern_op(compiler, op_name):
    """Get the extern op function from the registered compiler
    """
    if compiler in compilers:
        if hasattr(compilers[compiler], 'extern_op'):
            extern_op = getattr(compilers[compiler], 'extern_op')
            if hasattr(extern_op, op_name):
                return getattr(extern_op, op_name)

    logger.warning("%s in %s is not registered. Fallback to CPU" % (op_name, compiler))
    return lambda x, y: False

@reg.register_extern_op("nn.conv2d")
def external_conv2d(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'conv2d')(attrs, args)


@reg.register_extern_op("nn.dense")
def external_dense(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'dense')(attrs, args)

@reg.register_extern_op("nn.relu")
def external_relu(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'relu')(attrs, args)

@reg.register_extern_op("nn.batch_norm")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'batch_norm')(attrs, args)

@reg.register_extern_op("subtract")
def external_subtract(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'subtract')(attrs, args)

@reg.register_extern_op("add")
def external_add(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'add')(attrs, args)

@reg.register_extern_op("multiply")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'multiply')(attrs, args)
