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

from . import gcc
from .. import op as reg

@reg.register_extern_op("nn.conv2d")
def external_conv2d(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    if compiler == "gcc":
        return gcc.extern_op.conv2d(attrs, args)

    raise RuntimeError("conv2d in {} is not registered" % (compiler))


@reg.register_extern_op("subtract")
def external_subtract(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    if compiler == "gcc":
        return gcc.extern_op.subtract(attrs, args)

    raise RuntimeError("subtract in {} is not registered" % (compiler))


@reg.register_extern_op("add")
def external_add(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    if compiler == "gcc":
        return gcc.extern_op.add(attrs, args)

    raise RuntimeError("add in {} is not registered" % (compiler))


@reg.register_extern_op("multiply")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    if compiler == "gcc":
        return gcc.extern_op.multiply(attrs, args)

    raise RuntimeError("multiply in {} is not registered" % (compiler))
