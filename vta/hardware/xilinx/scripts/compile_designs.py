#!/usr/bin/env python
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
import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
import re
import time
from collections import namedtuple
from numpy import floor, ceil, log2, log10
from subprocess import call

FPGA = namedtuple("FPGAConstraints",
                  ['bram_w', 'bram_d', 'num_bram'])

Hardware = namedtuple("HWConfig",
                      ['batch', 'block_in', 'block_out',
                       'input_w', 'weight_w', 'accum_w', 'out_w', 'uop_w'])

def find_bram_confs(fpga, hw_conf, log_uop_sizeB):
  # Derive sizes
  input_elem_size_b = hw_conf.batch*hw_conf.block_in*hw_conf.input_w
  weight_elem_size_b = hw_conf.block_in*hw_conf.block_out*hw_conf.weight_w
  accum_elem_size_b = hw_conf.batch*hw_conf.block_out*hw_conf.accum_w
  input_min_bram = (input_elem_size_b+fpga.bram_w-1)/fpga.bram_w
  weight_min_bram = (weight_elem_size_b+fpga.bram_w-1)/fpga.bram_w
  accum_min_bram = (accum_elem_size_b+fpga.bram_w-1)/fpga.bram_w
  # Exploring all possible BRAM distributions
  bram_confs = []
  uop_bram = pow(2, log_uop_sizeB) * 8 / (fpga.bram_w * fpga.bram_d)
  for log_i_bram in range(int(log2(input_min_bram)), int(ceil(log2(fpga.num_bram)))):
    i_bram = pow(2, log_i_bram)
    for log_w_bram in range(int(log2(weight_min_bram)), int(ceil(log2(fpga.num_bram)))):
      w_bram = pow(2, log_w_bram)
      for log_a_bram in range(int(log2(accum_min_bram)), int(ceil(log2(fpga.num_bram)))):
        a_bram = pow(2, log_a_bram)
        total_bram = uop_bram + i_bram + w_bram + a_bram + a_bram / hw_conf.accum_w * hw_conf.out_w
        if total_bram <= fpga.num_bram:
          # Right now we need to restrict uop width
          input_elems = i_bram * fpga.bram_w * fpga.bram_d / input_elem_size_b
          weight_elems = w_bram * fpga.bram_w * fpga.bram_d / weight_elem_size_b
          accum_elems = a_bram * fpga.bram_w * fpga.bram_d / accum_elem_size_b
          if log2(input_elems) + log2(weight_elems) + log2(accum_elems) <= hw_conf.uop_w:
            log_inp_sizeB = int(log2(i_bram*fpga.bram_d*fpga.bram_w/8))
            log_wgt_sizeB = int(log2(w_bram*fpga.bram_d*fpga.bram_w/8))
            log_acc_sizeB = int(log2(a_bram*fpga.bram_d*fpga.bram_w/8))
            bram_confs.append([log_uop_sizeB, log_inp_sizeB, log_wgt_sizeB, log_acc_sizeB])
  # Filter out configs that are suboptimal
  suboptimal = [False] * len(bram_confs)
  for i in range(0, len(bram_confs)):
    for j in range(i + 1, len(bram_confs)):
      leq_list = [a <= b for a, b in zip(bram_confs[i], bram_confs[j])]
      geq_list = [a >= b for a, b in zip(bram_confs[i], bram_confs[j])]
      leq = all(leq_list)
      geq = all(geq_list)
      if leq:
        suboptimal[i] = True
      if geq:
        suboptimal[j] = True
  opt_bram_confs = [x[0] for x in zip(bram_confs, suboptimal) if not x[1]]
  return opt_bram_confs

def get_make_command(job, build_dir, hw_conf, bram_conf, mode, slurm=False):
  cmd = ""
  if slurm:
    cmd += "#!/bin/bash\n"
    cmd += "#SBATCH --job-name={}\n".format(job)
    cmd += "#SBATCH --output={}.out\n".format(job)
    cmd += "srun "
  if mode=="hls":
    cmd += "make ip"
  else:
    cmd += "make"
  cmd += " SLURM={} MODE=skip_sim NO_DSP=false NO_ALU=false".format("true" if slurm else "false")
  cmd += " BUILD_NAME={}".format(build_dir)
  cmd += " VTA_LOG_INP_WIDTH={}".format(int(log2(hw_conf.input_w)))
  cmd += " VTA_LOG_WGT_WIDTH={}".format(int(log2(hw_conf.weight_w)))
  cmd += " VTA_LOG_BATCH={}".format(int(log2(hw_conf.batch)))
  cmd += " VTA_LOG_BLOCK_IN={}".format(int(log2(hw_conf.block_in)))
  cmd += " VTA_LOG_BLOCK_OUT={}".format(int(log2(hw_conf.block_out)))
  cmd += " VTA_LOG_UOP_BUFF_SIZE={}".format(bram_conf[0])
  cmd += " VTA_LOG_INP_BUFF_SIZE={}".format(bram_conf[1])
  cmd += " VTA_LOG_WGT_BUFF_SIZE={}".format(bram_conf[2])
  cmd += " VTA_LOG_ACC_BUFF_SIZE={}\n".format(bram_conf[3])
  return cmd

def cli():
  parser = argparse.ArgumentParser(
      description='Analyze HLS experiments'
  )
  parser.add_argument(
      '-mode', dest='mode', action='store', type=str, required=True,
      choices=["hls", "vivado"], help='hls synthesis or full compilation'
  )
  parser.add_argument(
      '-base_dir', dest='base_dir', action='store', type=str, required=False,
      default="../../build/hardware/xilinx/", help='path to build directory'
  )
  parser.add_argument(
      '-min_ibw', dest='min_ibw', action='store', type=int, required=False,
      default=3, help='log2 of minimum input bit-width'
  )
  parser.add_argument(
      '-max_ibw', dest='max_ibw', action='store', type=int, required=False,
      default=3, help='log2 of maximum input bit-width'
  )
  parser.add_argument(
      '-min_wbw', dest='min_wbw', action='store', type=int, required=False,
      default=3, help='log2 of minimum weight bit-width'
  )
  parser.add_argument(
      '-max_wbw', dest='max_wbw', action='store', type=int, required=False,
      default=3, help='log2 of maximum weight bit-width'
  )
  parser.add_argument(
      '-acc_bw', dest='acc_bw', action='store', type=int, required=False,
      default=32, help='accumulator bit-width'
  )
  parser.add_argument(
      '-uop_bw', dest='uop_bw', action='store', type=int, required=False,
      default=32, help='micro-op bit-width'
  )
  parser.add_argument(
      '-min_batch', dest='min_batch', action='store', type=int, required=False,
      default=0, help='log2 of minimum batch size'
  )
  parser.add_argument(
      '-max_batch', dest='max_batch', action='store', type=int, required=False,
      default=8, help='log2 of maximum batch size'
  )
  parser.add_argument(
      '-min_ic', dest='min_ic', action='store', type=int, required=False,
      default=0, help='log2 of minimum input channels'
  )
  parser.add_argument(
      '-max_ic', dest='max_ic', action='store', type=int, required=False,
      default=8, help='log2 of maximum input channels'
  )
  parser.add_argument(
      '-min_oc', dest='min_oc', action='store', type=int, required=False,
      default=0, help='log2 of minimum output channels'
  )
  parser.add_argument(
      '-max_oc', dest='max_oc', action='store', type=int, required=False,
      default=8, help='log2 of maximum output channels'
  )
  parser.add_argument(
      '-uop_sizeB', dest='uop_sizeB', action='store', type=int, required=False,
      default=14, help='log2 of uop buffer in B'
  )
  parser.add_argument(
      '-bram_w', dest='bram_w', action='store', type=int, required=False,
      default=32, help='FPGA BRAM port width in b'
  )
  parser.add_argument(
      '-bram_d', dest='bram_d', action='store', type=int, required=False,
      default=1024, help='FPGA BRAM depth'
  )
  parser.add_argument(
      '-num_bram', dest='num_bram', action='store', type=int, required=False,
      default=124, help='FPGA total BRAM'
  )
  parser.add_argument(
      '-slurm', dest='slurm', action='store_true',
      help='Run on cluster using slurm'
  )
  args = parser.parse_args()

  # Logging
  logging.basicConfig(filename='compile_designs.log',level=logging.DEBUG)

  # FPGA config
  pynq = FPGA(args.bram_w, args.bram_d, args.num_bram)

  # Get timestamp
  timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
  build_dir = "build_{}".format(timestamp)

  num_confs = 0
  for log_ibw in range(args.min_ibw, args.max_ibw+1):
    ibw = pow(2, log_ibw)
    for log_wbw in range(args.min_wbw, args.max_wbw+1):
      wbw = pow(2, log_wbw)
      for log_batch in range(args.min_batch, args.max_batch+1):
        batch = pow(2, log_batch)
        for log_ic in range(args.min_ic, args.max_ic+1):
          ic = pow(2, log_ic)
          for log_oc in range(args.min_oc, args.max_oc+1):
            oc = pow(2, log_oc)
            conf = Hardware(batch, ic, oc, ibw, wbw, args.acc_bw, ibw, args.uop_bw)
            bram_confs = find_bram_confs(pynq, conf, args.uop_sizeB)
            for b in bram_confs:
              job = "{}x{}x{}_{}bx{}b_{}_{}_{}_{}_100MHz_10ns".format(
                batch, ic, oc, ibw, wbw, b[0], b[1], b[2], b[3])
              num_confs += 1
              cmd = get_make_command(job, build_dir, conf, b, args.mode, args.slurm)
              sb_file = job+".sb"
              file = open(sb_file,"w")
              file.write(cmd)
              file.close()
              call(["echo", cmd])
              if args.slurm:
                call(["sbatch", sb_file])
              else:
                call(cmd.split(" "))

if __name__ == '__main__':
  cli()
