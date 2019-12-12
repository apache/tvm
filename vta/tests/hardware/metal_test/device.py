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

from ctypes import *
from _macros_h import *
from insn_lib import *
import numpy as np

# dictionary that maps actions to their corresponding value
ACTIONS = {"1DLOAD":VTA_OPCODE_LOAD, 
           "2DLOAD":VTA_OPCODE_LOAD, 
           "ALU":None, 
           "GEMM":None, 
           "1DSTORE":VTA_OPCODE_STORE, 
           "2DSTORE":VTA_OPCODE_STORE, 
           "FINISH":None}

# dictionary that maps items to their corresponding ID
ITEMS = {"UOP":VTA_MEM_ID_UOP, 
         "INP":VTA_MEM_ID_INP, 
         "WGT":VTA_MEM_ID_WGT, 
         "ACC":VTA_MEM_ID_ACC, 
         "OUT":VTA_MEM_ID_OUT,
# alu opcodes to their corresponding value
         "MIN":VTA_ALU_OPCODE_MIN, 
         "MAX":VTA_ALU_OPCODE_MAX, 
         "ADD":VTA_ALU_OPCODE_ADD, 
         "SHR":VTA_ALU_OPCODE_SHR, 
         "EMPTY":None}

class InsnStream:

	def __init__(self):
		self.insn_buf = []
	
	"""
	Add a instruction to the instruction stream
		Args:
			action: Instruction type (ignores case)
			item: Item or ALU opcode for ALU actions (ignores case)
			others: micro-ops for different instructions

		Raises:	
			ValueError: If action or item is invalid
	"""
	def add(self, action, item="EMPTY", sram=0, dram=0, use_imm=False, imm_val=0, 
	 y_size=1, x_size=0, x_stride=0, x_pad=0, y_pad=0, vec_len=0):
		if (action.upper() not in ACTIONS) or (item.upper() not in ITEMS):
			raise ValueError("Invalid opcode/item")
		self.insn_buf.append((action.upper(), item.upper(), sram, dram, use_imm, imm_val, y_size, x_size, x_stride, x_pad, y_pad, vec_len))
	
	"""Clear the stream"""
	def clear(self):
		self.insn_buf = []

class Device:

	def __init__(self, batch, in_channels, out_channels, insn_stream, uop_compression=True):

		self.insn_stream = insn_stream
		# set up dimensions
		self.batch, self.in_channels, self.out_channels = batch, in_channels, out_channels
		
		# set up flags
		self.uop_compression = uop_compression

		# gemm size
		self.ins_size = len(insn_stream.insn_buf)
		self.inp_size = (int)(batch / VTA_BATCH * in_channels / VTA_BLOCK_IN)
		self.wgt_size = (int) (in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT)
		self.out_size = (int) (batch / VTA_BATCH * out_channels / VTA_BLOCK_OUT)

		# get every computing stage and set uop_size buffer accordingly
		self.uop_size_buf = []
		self.action_buf = []
		# loop through instruction
		for i in range(self.ins_size):
			insn = self.insn_stream.insn_buf[i]
			# if instruction is GEMM or ALU, save to action buffer and uop size for that instruction to uop size buffer
			if insn[0] == "GEMM" or insn[0] == "ALU":
				if insn[0] == "GEMM":
					uop_size = (int) (batch / VTA_BATCH) if uop_compression else \
				(int) (batch / VTA_BATCH * in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT)
				elif insn[0] == "ALU":
					uop_size = 1 if uop_compression else out_channels // VTA_BLOCK_OUT
				self.uop_size_buf.append(uop_size)
				self.action_buf.append(insn[0])
		
		# assert dimensions is divisible
		assert batch % VTA_BATCH == 0
		assert in_channels % VTA_BLOCK_IN == 0
		assert out_channels % VTA_BLOCK_OUT == 0

		# generate instructions
		self.__generate_insn()

		# uop setup
		self.__uop_setup()

	"""print the instructions for this device"""
	def print_insn(self):
		print_instruction(self.ins_size, self.insn_buf)
	"""print the micro-ops for this device"""
	def print_uop(self):
		print_uop(np.sum(self.uop_size_buf), self.uop_buf)
	"""print the parameters for this device"""
	def print_params(self):
		print_params()

	def __generate_insn(self):
		# previous state, present state and next state
		prevs, ps, ns = "load", "load", "load"
		self.insn_buf = (VTAGenericInsn * self.ins_size)()
		index, cmp_ind = 0, 0
		assert "LOAD" in self.insn_stream.insn_buf[0][0], "Instruction shold start with Load instruction"
		SIZE = {"INP":self.inp_size, "WGT":self.wgt_size, "ACC":self.out_size, "OUT":self.out_size}

		# generate instructions
		for i in range(self.ins_size):
			# check for order
			if (i + 1) < self.ins_size: 
				n_action, n_item = self.insn_stream.insn_buf[i+1][0], self.insn_stream.insn_buf[i+1][1]
				# check for order
				if "LOAD" in n_action:
					ns = "load"
				elif n_action == "GEMM" or n_action == "ALU":
					ns = "compute"
				elif "STORE" in n_action:
					ns = "store"
				elif n_action == "FINISH":
					ns = "finish"

				assert not(ps == "load" and ns == "store"), "Instruction Wrong order!"
				assert not(ps == "compute" and ns == "load"), "Instruction Wrong order!"
				assert not(ps == "store" and ns == "compute"), "Instruction Wrong order!"

			# generate instructions for GEMM 
			action, item, sram, dram, use_imm, imm_val, y_size, x_size, \
			 x_stride, x_pad, y_pad, vec_len = self.insn_stream.insn_buf[i]
			# size
			size = SIZE.get(item) if (item != "UOP") else self.uop_size_buf[cmp_ind]

			# dependencies
			pop_next = 1 if (prevs == "store" and ps == "load") else 0
			push_next = 1 if (ps == "load" and ns == "compute") else 0

			# make sure dependancies, ACC don't need push_next dependencies
			# generate instructions based on different instructions in the buffer and add them into instruction buffer
			if action == "1DLOAD":
				self.insn_buf[index] = get_1Dloadstore_insn(ACTIONS.get(action), ITEMS.get(item), sram, dram, size, 
				 0, pop_next, 0, push_next)

			elif action == "2DLOAD":
				self.insn_buf[index] = get_2Dloadstore_insn(ACTIONS.get(action), ITEMS.get(item), sram, dram, 
				 y_size, x_size, x_stride, y_pad, x_pad, 0, pop_next, 0, 0)

			elif action == "GEMM":
				self.insn_buf[index] = get_gemm_insn(0, (int) (self.batch / VTA_BATCH), (int) (self.in_channels / VTA_BLOCK_IN),
				 (int) (self.out_channels / VTA_BLOCK_OUT), self.uop_compression, 1, 0, 0, 1)
				cmp_ind += 1
			# ALU opcodes are viewed as items
			elif action == "ALU":
				self.insn_buf[index] = get_alu_insn(ITEMS.get(item), vec_len, use_imm, imm_val, self.uop_compression, 
				 0, 0, 0, 1)
				cmp_ind += 1

			elif action == "1DSTORE":
				self.insn_buf[index] = get_1Dloadstore_insn(ACTIONS.get(action), ITEMS.get(item), sram, dram, size, 
				 1, 0, 1, 0)

			elif action == "2DSTORE":
				self.insn_buf[index] = get_2Dloadstore_insn(ACTIONS.get(action), ITEMS.get(item), sram, dram, 
				 y_size, x_size, x_stride, y_pad, x_pad, 1, 0, 1, 0)

			elif action == "FINISH":
				self.insn_buf[index] = get_finish_insn(0, 1)
			index += 1
			prevs = ps
			ps = ns
		assert ns == "finish", "Did not end with finish instruction"
		
	def __pointer_setup(self, code, tensor):
		if code == 0:
			# input
			el_size, dim1, dim2 = VTA_INP_ELEM_BYTES, VTA_BATCH, VTA_BLOCK_IN
		elif code == 1:
			# weight
			el_size, dim1, dim2 = VTA_WGT_ELEM_BYTES, VTA_BLOCK_OUT, VTA_BLOCK_IN
		elif code == 2:
			# acc
			el_size, dim1, dim2 = VTA_ACC_ELEM_BYTES, VTA_BATCH, VTA_BLOCK_OUT
		row, clmn = tensor.shape
		size = (int) (row / dim1 * clmn / dim2)
		pointer = alloc_buffer((int) (el_size * size))
		res_pointer = cast(pointer, POINTER(c_uint32))
		pack_buffer(code, res_pointer, get_2dpointer(tensor), row, clmn, dim1, dim2)
		return res_pointer

	def __uop_setup(self):
		self.uop_buf = alloc_buffer(sizeof(VTAUop) * np.sum(self.uop_size_buf))
		self.uop_buf = cast(self.uop_buf, POINTER(VTAUop))
		cur_size = 0
		# for each action, add corresponding uop
		for i in range(len(self.action_buf)):
			action = self.action_buf[i]
			if action == "GEMM":
				item = get_gemm_uops(
					self.batch // VTA_BATCH,
					self.in_channels // VTA_BLOCK_IN,
					self.out_channels // VTA_BLOCK_OUT,
					self.uop_compression,
					0)
			elif action == "ALU":
				item = get_map_alu_uops(self.out_channels // VTA_BLOCK_OUT, self.uop_compression)
			concat_uop(self.uop_buf, item, cur_size, self.uop_size_buf[i])
			cur_size += self.uop_size_buf[i]

	def run(self, input, weight, bias, output_ref):
		# make sure shape matches
		assert self.batch == input.shape[0]
		assert self.in_channels == input.shape[1]
		assert self.out_channels == weight.shape[0]

		# tensor pointer setup
		input_pointer = self.__pointer_setup(0, input.astype(inp_T))
		weight_pointer = self.__pointer_setup(1, weight.astype(wgt_T))
		bias_pointer = self.__pointer_setup(2, bias.astype(acc_T))

		# prepare output buffer
		output_buf = alloc_buffer((int) (VTA_OUT_ELEM_BYTES * self.out_size))
		output_pointer = cast(output_buf, POINTER(c_uint32))

		# insn pointer setup
		size = sizeof(VTAGenericInsn) * self.ins_size
		insn_pointer = alloc_buffer(size)
		_insn_pointer = cast(insn_pointer, POINTER(VTAGenericInsn))
		transfer(_insn_pointer, c_void_p(addressof(self.insn_buf)), size)

		# call fpga
		t_fpga = vta_run(self.ins_size, _insn_pointer, self.uop_buf, input_pointer, weight_pointer, bias_pointer, output_pointer)
		time = t_fpga / 1E6
		ops = 0
		for action in self.action_buf:
			inc = self.batch * self.in_channels *self.out_channels * 2 if action == "GEMM" else self.out_channels
			ops += inc
		gops = ops / t_fpga

		# get computed output
		outputs = alloc_2darray(self.batch, self.out_channels)
		cast(outputs, POINTER(POINTER(out_T)))
		unpack_buffer(outputs, output_pointer, self.batch, self.out_channels, VTA_BATCH, VTA_BLOCK_OUT)
	
		# convert output to np array
		outputs_np = get_2darray(outputs, self.batch, self.out_channels)
		# test
		np.testing.assert_array_equal(output_ref.astype(out_T), outputs_np)
		# print results
		print("Time is {:.4} ms".format(time))
		print("Throughput is {:.4} Gops/s \n".format(gops))
		return (time, gops)
