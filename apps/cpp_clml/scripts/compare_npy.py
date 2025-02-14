#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import sys
import numpy as np


def main():
    print("Compare given numpy array in npz files")
    if len(sys.argv) != 4:
        print("Usage: python compare_npy.py <npz file 1> <npz file 2> <np array to cpmpare>")
        return

    in1 = np.load(sys.argv[1])
    in2 = np.load(sys.argv[2])

    print(sys.argv[1] + "->" + sys.argv[3] + ":", in1[sys.argv[3]].shape)
    print(sys.argv[2] + "->" + sys.argv[3] + ":", in1[sys.argv[3]].shape)

    np.testing.assert_allclose(in1[sys.argv[3]], in2[sys.argv[3]], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
