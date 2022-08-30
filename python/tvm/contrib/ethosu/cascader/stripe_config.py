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
"""Stripe config class to hold tensor striping information."""
# pylint: disable=invalid-name
import tvm._ffi

from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("contrib.ethosu.cascader.StripeConfig")
class StripeConfig(Object):
    """StripeConfig class"""

    def __init__(self, shape, extent, strides, order, stripes, offset):
        strides = list([float(v) for v in strides])
        self.__init_handle_by_constructor__(
            _ffi_api.StripeConfig, shape, extent, strides, order, stripes, offset
        )

    @property
    def shape(self):
        return list(self._shape)

    @property
    def extent(self):
        return list(self._extent)

    @property
    def strides(self):
        return list([float(v.value) for v in self._strides])

    @property
    def order(self):
        return list(self._order)

    @property
    def stripes(self):
        return list(self._stripes)

    @property
    def offset(self):
        return list(self._offset)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return _ffi_api.StripeConfigEqual(self, other)

    def __repr__(self):
        return (
            f"StripeConfig(shape={self.shape}, "
            f"extent={self.extent}, "
            f"strides={self.strides}, "
            f"order={self.order}, "
            f"stripes={self.stripes}, "
            f"offset={self.offset}"
        )


def count_stripes(stripe_config: StripeConfig, enable_sliding_window: bool = False):
    stripe_counts = dict(_ffi_api.CountStripes(stripe_config, enable_sliding_window))
    # Some code to 'de-TVM' the data types and make them pure Python
    clean_stripe_counts = dict()
    for stripe, count in stripe_counts.items():
        clean_stripe = tuple([int(v) for v in stripe])
        clean_count = int(count)
        clean_stripe_counts[clean_stripe] = clean_count

    return clean_stripe_counts
