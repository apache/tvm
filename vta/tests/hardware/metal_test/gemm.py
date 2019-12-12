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

def main():
	# tensor setup
	batch, in_channels, out_channels = 256, 16, 16
	input_buf = np.random.randint(RAND_MAX, size=(batch, in_channels))
	weight_buf = np.random.randint(RAND_MAX, size=(out_channels, in_channels))
	bias_buf = np.random.randint(RAND_MAX, size=(batch, out_channels))

	# create instructions
	istream = InsnStream()
	istream.add("1DLOAD", "UOP", sram=0, dram=0)
	istream.add("1DLOAD", "ACC", sram=0, dram=0)
	istream.add("1DLOAD", "WGT", sram=0, dram=0)
	istream.add("1DLOAD", "INP", sram=0, dram=0)
	istream.add("GEMM")
	istream.add("1DSTORE", "OUT", sram=0, dram=0)
	istream.add("FINISH")

	# compute reference function
	output_ref = np.zeros((batch, out_channels)).astype(out_T)
	np.seterr(over='ignore')
	for i in range(batch):
		for j in range(out_channels):
			sum = bias_buf[i][j]
			for k in range(in_channels):
				sum = sum + input_buf[i][k] * weight_buf[j][k]
			output_ref[i][j] = sum

	# run device
	dev = Device(batch, in_channels, out_channels, istream, uop_compression=True)
	dev.run(input_buf, weight_buf, bias_buf, output_ref)

if __name__ == 'main':
	main()