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

from numpy import isclose
import random
from tvm.autotvm import util


SI_PREFIXES = 'yzafpn\xb5m kMGTPEZY'


def test_format_si_prefix():
  # test float conversion
  assert util.format_si_prefix(1024, 'k') == 1.024

  for i, prefix in enumerate(SI_PREFIXES):
    integer, decimal = random.randint(0, 1000), random.randint(0, 1000)
    exp = -24 + 3 * i   # 0th prefix (yocto) is 10^-24
    number = integer * (10 ** exp) + decimal * (10 ** (exp - 3))
    expected = (integer + decimal / 1000)
    assert isclose(util.format_si_prefix(number, prefix), expected)

  assert util.format_si_prefix(0, 'y') == 0


if __name__ == '__main__':
  test_format_si_prefix()
