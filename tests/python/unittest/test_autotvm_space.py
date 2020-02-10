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
"""Test space definition primitives"""

import tvm
from tvm.autotvm.task.space import ConfigSpace, FallbackConfigEntity

def gemm_func(cfg, N):
    A = tvm.placeholder((N, N), name='A')
    B = tvm.placeholder((N, N), name='B')

    k = tvm.reduce_axis((0, N), name='k')
    C = tvm.compute((N, N), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=[k]), name='C')

    s = tvm.create_schedule([C.op])

    y, x = s[C].op.axis

    cfg.define_split('tile_y', cfg.axis(y), num_outputs=2)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=2)

    return s, [A, B, C]

def test_split():
    cfg = ConfigSpace()

    gemm_func(cfg, 128)
    assert len(cfg) == 64
    assert len(cfg.space_map['tile_y']) == 8

    # test policy
    cfg = ConfigSpace()
    cfg.define_split('tile_x', cfg.axis(256), policy='factors', num_outputs=3)
    assert len(cfg.space_map['tile_x']) == 45

    cfg.define_split('tile_y', cfg.axis(256), policy='power2', num_outputs=3)
    assert len(cfg.space_map['tile_y']) == 45

    cfg.define_split('tile_z', cfg.axis(256), policy='verbose', num_outputs=3)
    assert len(cfg.space_map['tile_z']) == 45

    cfg.define_split('tile_a', cfg.axis(224), policy='factors', num_outputs=3)
    assert len(cfg.space_map['tile_a']) == 63

    cfg.define_split('tile_b', cfg.axis(224), policy='power2', num_outputs=3)
    assert len(cfg.space_map['tile_b']) == 36

    cfg.define_split('tile_c', cfg.axis(224), policy='verbose', num_outputs=3)
    assert len(cfg.space_map['tile_c']) == 84

    # test fallback
    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(128), num_outputs=3)
    cfg.fallback_split('tile_n', [-1, 8, 4])
    assert cfg['tile_n'].size == [4, 8, 4]

    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(49), num_outputs=3)
    cfg.fallback_split('tile_n', [-1, 8, 4])
    assert cfg['tile_n'].size == [7, 7, 1]

    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(49), num_outputs=3)
    try:
        cfg.fallback_split('tile_n', [-1, 1, 0])
        assert False
    except RuntimeError:
        pass


if __name__ == '__main__':
    test_split()
