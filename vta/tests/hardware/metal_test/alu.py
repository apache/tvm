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

from device import *
import numpy as np

# tensor setup
vec_size, batch = 16, 256
alu_vec = np.random.randint(RAND_MAX, size=(batch,vec_size)).astype(acc_T)
output_ref = alu_vec << 1
istream = InsnStream()
tx_size = vec_size // VTA_BLOCK_OUT

# create instructions
istream.add("1DLOAD", "UOP")
for b in range(0, batch, VTA_BATCH):
	use_imm = True
	input_sets = 1 if use_imm else 2

	istream.add("2DLOAD", "ACC", sram=0, dram=(int)(b / VTA_BATCH * tx_size * input_sets), 
	 y_size=1, x_size=(int) (tx_size * input_sets), x_stride=(int) (tx_size * input_sets), y_pad=0, x_pad=0)

	istream.add("ALU", "SHR", use_imm=use_imm, imm_val=-1, vec_len=tx_size)

	istream.add("2DSTORE", "OUT", sram=0, dram=(int)(b / VTA_BATCH * tx_size), 
	 y_size=1, x_size=tx_size, x_stride=tx_size, y_pad=0, x_pad=0)
istream.add("FINISH")

# run device
dev = Device(batch, vec_size, vec_size, istream, uop_compression=True)
dev.run(np.zeros((batch, vec_size)), np.zeros((vec_size, vec_size)), alu_vec, output_ref)
