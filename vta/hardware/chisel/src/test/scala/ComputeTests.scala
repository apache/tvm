// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class ComputeTests(c: Compute)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  //==============================
  // ALU Instructions
  //==============================
  // min
  val insn_min_uop = "h00000001000100010000000000000000".U // uop fetch (uop_compress=true)
  val insn_min_acc = "h00000008000800010000000000000180".U // acc fetch
  val insn_min_out = "h7ffe4002000008000010000800200044".U // out
  // min (rebase)
  val insn_min_rebase_acc = "h00000008000800010000000010000190".U // acc fetch
  // min (uop_compress=false)
  val insn_min_nocomp_uop = "h00000008000800010000000000000000".U // uop fetch (uop_compress=false)
  val insn_min_nocomp_acc = "h00000008000800010000000000000180".U // acc fetch
  val insn_min_nocomp_out = "h0002c000000000000002000801000044".U // out
  // max
  val insn_max_uop = "h00000001000100010000000000000000".U // uop fetch (uop_compress=true)
  val insn_max_acc = "h00000008000800010000000000000180".U // acc fetch
  val insn_max_out = "h00015002000008000010000800200044".U // out
  // shr
  val insn_shr_uop = "h00000001000100010000000000000000".U // uop fetch
  val insn_shr_acc = "h00000008000800010000000000000180".U // acc fetch
  val insn_shr_out = "h00017002000008000010000800200044".U // out
  // finish
  val insn_finish = "h00000000000000000000000000000013".U // finish

  //==============================
  // ALU Inputs and Outputs
  //==============================
  // min
  val biases_min_data0 = "fffffff6ffffffc4ffffffc7ffffffefffffffc1ffffffecfffffff30000001e"+
                         "000000040000000a0000000b00000039ffffffeeffffffc6ffffffe9fffffff5"
  val biases_min_data1 = "000000120000003afffffffb00000010fffffff3ffffffde00000034fffffffa"+
                         "0000002a00000037ffffffcd00000022ffffffd3ffffffc60000003400000020"
  val biases_min_data2 = "000000310000003efffffffcffffffd800000031000000170000000b0000003e"+
                         "ffffffd10000002200000038ffffffd0ffffffdbfffffff8ffffffc7ffffffc7"
  val biases_min_data3 = "ffffffe20000003c00000023ffffffddffffffdb00000016000000200000000f"+
                         "0000002bffffffd9ffffffc5fffffff800000007fffffff8ffffffec00000031"
  val biases_min_data4 = "fffffff8ffffffe10000000afffffff30000001400000017ffffffdcffffffd3"+
                         "00000028000000290000002e0000000f00000018000000250000002bffffffe3"
  val biases_min_data5 = "000000020000001b00000009000000300000003b00000016ffffffe90000002d"+
                         "fffffffa0000001fffffffe9ffffffca0000001000000019ffffffce00000021"
  val biases_min_data6 = "ffffffd3000000160000003affffffe7fffffff2ffffffd20000002e00000003"+
                         "0000001100000008000000320000001d0000002f00000034ffffffdefffffff1"
  val biases_min_data7 = "fffffffa0000003ffffffff50000002f00000019ffffffc8ffffffd6fffffffa"+
                         "0000001ffffffff1000000010000003efffffff6000000110000002400000017"
  val biases_min = IndexedSeq(biases_min_data0, biases_min_data1, biases_min_data2, biases_min_data3,
                              biases_min_data4, biases_min_data5, biases_min_data6, biases_min_data7)

  // min (uop_compress=false)
  val biases_min_nocomp_data0 = "00000001fffffff3ffffffe2ffffffc800000024ffffffdf00000012ffffffde"+
                                "00000025ffffffe8ffffffe9ffffffdb0000002100000010ffffffd6ffffffd5"
  val biases_min_nocomp_data1 = "ffffffe6ffffffe4ffffffd9ffffffee00000008ffffffe60000003fffffffe1"+
                                "ffffffd3000000020000000a000000270000000fffffffd0fffffffffffffffb"
  val biases_min_nocomp_data2 = "ffffffddffffffd0fffffff6ffffffd1ffffffd8000000330000002afffffff0"+
                                "00000032ffffffe7ffffffceffffffed00000021ffffffdf0000003affffffe3"
  val biases_min_nocomp_data3 = "00000038ffffffe4ffffffd3ffffffc500000037000000020000003cfffffff1"+
                                "00000036ffffffe5ffffffec0000002200000018000000190000001000000012"
  val biases_min_nocomp_data4 = "000000090000000cffffffc8ffffffdf00000004ffffffd00000001e00000009"+
                                "0000000e00000009ffffffddfffffffbfffffff60000001bffffffc9ffffffce"
  val biases_min_nocomp_data5 = "00000020fffffff5ffffffeffffffff5ffffffe1ffffffda0000003a0000001d"+
                                "0000002b00000020ffffffdcffffffecfffffffbffffffc4ffffffef0000001c"
  val biases_min_nocomp_data6 = "0000000d0000000c000000200000001b0000002f0000001d00000037ffffffd3"+
                                "0000003f00000036ffffffe10000002b000000280000002fffffffcaffffffc0"
  val biases_min_nocomp_data7 = "00000023fffffffffffffff5000000260000000effffffd6000000000000000e"+
                                "0000003a0000001a000000250000002d0000003f00000039ffffffe4ffffffc0"
  val biases_min_nocomp = IndexedSeq(biases_min_nocomp_data0, biases_min_nocomp_data1,
                                     biases_min_nocomp_data2, biases_min_nocomp_data3,
                                     biases_min_nocomp_data4, biases_min_nocomp_data5,
                                     biases_min_nocomp_data6, biases_min_nocomp_data7)

  // max
  val biases_max_data0 = "0000001a0000000b0000003600000004ffffffdf00000035fffffff200000003"+
                         "ffffffd4ffffffe1ffffffdcfffffffeffffffc300000028000000380000002e"
  val biases_max_data1 = "ffffffd8ffffffe5ffffffc9ffffffc5ffffffd6ffffffd000000033ffffffd0"+
                         "0000003afffffff5ffffffd6ffffffe20000003bffffffc50000000900000014"
  val biases_max_data2 = "fffffff9ffffffe60000001c00000017ffffffdaffffffed0000001b0000000f"+
                         "ffffffc3000000210000002a00000035000000190000000dfffffff4ffffffc2"
  val biases_max_data3 = "00000010fffffffc000000070000000fffffffcd00000009ffffffd4ffffffe6"+
                         "00000021fffffff2ffffffd20000002cffffffdc0000001effffffc3fffffffc"
  val biases_max_data4 = "ffffffecffffffd30000002400000003ffffffd0ffffffe1ffffffc500000038"+
                         "ffffffc500000037000000060000003bffffffc7ffffffd4ffffffffffffffc7"
  val biases_max_data5 = "00000020ffffffd9ffffffcb00000007ffffffc4fffffff2000000190000002b"+
                         "ffffffdffffffffcffffffc100000018ffffffdaffffffcdfffffff1ffffffea"
  val biases_max_data6 = "fffffffc0000003900000016ffffffefffffffcafffffff90000003800000024"+
                         "ffffffe00000000d0000003b00000038ffffffd7ffffffe500000021ffffffe7"
  val biases_max_data7 = "00000010ffffffe3ffffffddffffffd3ffffffc2fffffff3ffffffcbffffffc8"+
                         "fffffffafffffff9ffffffedffffffcf0000003e0000003bffffffd900000005"
  val biases_max = IndexedSeq(biases_max_data0, biases_max_data1, biases_max_data2, biases_max_data3,
                              biases_max_data4, biases_max_data5, biases_max_data6, biases_max_data7)

  // shr biases
  val biases_shr_data0 = "ffffffd6ffffffc1ffffffd6ffffffeaffffffdfffffffdbffffffc500000010"+
                         "00000037fffffffa000000020000002100000000006646c000007ffff6cd4878"
  val biases_shr_data1 = "00000019ffffffc1ffffffe4ffffffc5fffffff700000012ffffffc600000030"+
                         "fffffffd00000033ffffffc5ffffffc400000012ffffffe90000001fffffffc4"
  val biases_shr_data2 = "0000003bffffffd7ffffffc4ffffffd800000007ffffffeafffffff100000019"+
                         "ffffffc2ffffffd5ffffffd5ffffffdd0000003e00000035ffffffc4ffffffc8"
  val biases_shr_data3 = "0000000d0000003100000010fffffff900000031000000200000002f00000030"+
                         "fffffff50000002bffffffecffffffe0ffffffdb00000008fffffff0ffffffcf"
  val biases_shr_data4 = "000000200000003b000000210000003cffffffd5000000310000002a0000001a"+
                         "ffffffcaffffffc30000000300000004ffffffea00000000ffffffeb0000001f"
  val biases_shr_data5 = "ffffffc5000000220000000efffffff700000016ffffffdbffffffc90000003c"+
                         "00000030ffffffeaffffffd30000003c0000002dfffffffa0000003f0000003d"
  val biases_shr_data6 = "0000000dffffffd4000000330000003dffffffd300000019000000370000003b"+
                         "ffffffd80000002dffffffd50000003e000000230000001200000034ffffffed"
  val biases_shr_data7 = "ffffffcafffffffd00000026ffffffe50000002e0000002affffffdcfffffffc"+
                         "00000035ffffffd9ffffffc1fffffffe0000001000000026ffffffd400000034"
  val biases_shr = IndexedSeq(biases_shr_data0, biases_shr_data1, biases_shr_data2, biases_shr_data3,
                              biases_shr_data4, biases_shr_data5, biases_shr_data6, biases_shr_data7)

  // shr outputs
  val outputs_shr_data0 = "hf5f0f5faf7f6f1040dfe000800b0ff1e".U
  val outputs_shr_data1 = "h06f0f9f1fd04f10cff0cf1f104fa07f1".U
  val outputs_shr_data2 = "h0ef5f1f601fafc06f0f5f5f70f0df1f2".U
  val outputs_shr_data3 = "h030c04fe0c080b0cfd0afbf8f602fcf3".U
  val outputs_shr_data4 = "h080e080ff50c0a06f2f00001fa00fa07".U
  val outputs_shr_data5 = "hf10803fd05f6f20f0cfaf40f0bfe0f0f".U
  val outputs_shr_data6 = "h03f50c0ff4060d0ef60bf50f08040dfb".U
  val outputs_shr_data7 = "h03f50c0ff4060d0ef60bf50f08040dfb".U
  val outputs_shr = IndexedSeq(outputs_shr_data0, outputs_shr_data1, outputs_shr_data2, outputs_shr_data3,
                               outputs_shr_data4, outputs_shr_data5, outputs_shr_data6, outputs_shr_data7)

  //==============================
  // GEMM Instructions
  //==============================
  // blocked gemm
  val insn_blocked_gemm_uop = "h00000080008000010000000000000000".U // uop fetch
  val insn_blocked_gemm_acc = "h00000010000400400000000000000180".U // acc fetch
  val insn_blocked_gemm_out = "h0100100000400800000800200800002a".U // out

  //==============================
  // GEMM Inputs and Outputs
  //==============================
  val uops_blocked_gemm_data0 = "h0000600c000040080000200400000000".U // uops[0~3]
  val uops_blocked_gemm_data1 = "h0000e01c0000c0180000a01400008010".U // uops[4~7]
  val uops_blocked_gemm_data2 = "h0001602c000140280001202400010020".U // uops[...]
  val uops_blocked_gemm_data3 = "h0001e03c0001c0380001a03400018030".U // uops[...]
  val uops_blocked_gemm_data = IndexedSeq(uops_blocked_gemm_data0, uops_blocked_gemm_data1,
                                          uops_blocked_gemm_data2, uops_blocked_gemm_data3)

  val biases_blocked_gemm_data0 = "2b543790f8490793c86f73fb0ea27042de47061aef99d4fc2654438fca28f6cb"+
                                  "2a94845bdd2e1531da0619990b425147f55b7b8bd1226cdecb16ee032db07b57"
  val biases_blocked_gemm_data1 = "fd6689e1d25d9ca933126e07f8d4d364fb80e61ffd218fa9e104a76e112c9512"+
                                  "0d76d66e035853c13b4c9c990d41477002a01bb4f84a07b51a3677c8c850ed42"
  val biases_blocked_gemm_data2 = "e050ef112c7d4f55095c0866346e413dc2d4741d00785477d1612777f8ceb521"+
                                  "1460e4812ae647b83310a8e732c85a0f2ab2008eee09aeb7f383df0bc93c50ac"
  val biases_blocked_gemm_data3 = "c69d34f1d8d31dc51631bdb0da3ae2a43bc7d3b52cec72233f9d7693c2d8b0de"+
                                  "ebe91103381d7565168836bc36a22d18cf2bc45917c224c0c90b62553402809a"
  val biases_blocked_gemm_data4 = "eedc3012c5d509e5d0d7fd3eff19da2df9d8034811c7272bdf9c486bed16fbee"+
                                  "0c9ca9252b4e60d3e6a82f11cfac6642dd7ef41636cf346ef8d82ceec8b669d0"
  val biases_blocked_gemm_data5 = "e395bbc5fdd42162dc323b2a33dd4c2d1cb923f71a6cb5cad17f59693168e7b6"+
                                  "3b0000d920b491d0d0848f9f3097a500cf442f86e857a01ecd047ae0030de6d3"
  val biases_blocked_gemm_data6 = "fb68a21c270c5daa0772e44c054a5ebacfd645a20e0c6bfe215756b50580e35b"+
                                  "07ce6ed0d0c68be92d3046df085792872a1afd2b25cd29ecfba978f31ac1dfe9"
  val biases_blocked_gemm_data7 = "d8eabfe603c4cbe9cdab7e3efc3931a8cae17eeb10107f85c723f6381b5453c2"+
                                  "cf5a447a0fb5ce6b01cd440a01a2d1ceff79f645c47c93b626cf60b103ce3716"
  val biases_blocked_gemm = IndexedSeq(biases_blocked_gemm_data0, biases_blocked_gemm_data1,
                                       biases_blocked_gemm_data2, biases_blocked_gemm_data3,
                                       biases_blocked_gemm_data4, biases_blocked_gemm_data5,
                                       biases_blocked_gemm_data6, biases_blocked_gemm_data7)

  def init_compute_min() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, "h4000".U)
  step(1)
  poke(c.io.gemm_queue.data, insn_min_uop)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)

  poke(c.io.uops.waitrequest, 1.U)
  poke(c.io.gemm_queue.data, insn_min_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  step(1)
  poke(c.io.uops.waitrequest, 0.U)
  expect(c.io.uops.read, 1.U)
  step(1)
  poke(c.io.uops.waitrequest, 1.U)
  step(1)
  poke(c.io.uops.waitrequest, 0.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  expect(c.io.uops.read, 0.U)
  step(1)
  expect(c.io.uops.read, 0.U)
  step(1)
  }

  def init_compute_min_rebase() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 1.U)
  poke(c.io.gemm_queue.data, insn_min_rebase_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  step(1)
  expect(c.io.biases.read, 0.U)
  step(1)

  expect(c.io.s2g_dep_queue.ready, 0.U)
  poke(c.io.s2g_dep_queue.data, 1.U)
  poke(c.io.s2g_dep_queue.valid, 1.U)
  step(1)
  expect(c.io.s2g_dep_queue.ready, 1.U)
  poke(c.io.s2g_dep_queue.valid, 0.U)
  step(1)
  }
  
  def init_compute_min_nocomp() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, "h00005803000050020000480100004000".U)
  step(1)
  poke(c.io.gemm_queue.data, insn_min_nocomp_uop)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  expect(c.io.uops.address, "h0000".U)
  step(1)
  expect(c.io.uops.address, "h0010".U)
  poke(c.io.uops.readdata, "h00007807000070060000680500006004".U)
  step(1)
  step(1)
  step(1)

  poke(c.io.gemm_queue.data, insn_min_nocomp_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  }

  def init_compute_max() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, "h00005803000050020000480100004000".U)
  step(1)
  poke(c.io.gemm_queue.data, insn_max_uop)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  expect(c.io.uops.address, "h0000".U)
  step(1)
  expect(c.io.uops.address, "h0010".U)
  poke(c.io.uops.readdata, "h00007807000070060000680500006004".U)
  step(1)
  step(1)
  step(1)

  poke(c.io.gemm_queue.data, insn_max_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  }


  def init_compute_shr() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, "h00005803000050020000480100004000".U)
  step(1)
  poke(c.io.gemm_queue.data, insn_shr_uop)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  expect(c.io.uops.address, "h0000".U)
  step(1)
  expect(c.io.uops.address, "h0010".U)
  poke(c.io.uops.readdata, "h00007807000070060000680500006004".U)
  step(1)
  step(1)
  step(1)

  poke(c.io.gemm_queue.data, insn_shr_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  }
  
  def test_compute_finish() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 1.U)
  poke(c.io.gemm_queue.data, insn_finish)
  poke(c.io.gemm_queue.valid, 1.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 0.U)
  step(1)

  expect(c.io.s2g_dep_queue.ready, 0.U)
  poke(c.io.s2g_dep_queue.data, 1.U)
  poke(c.io.s2g_dep_queue.valid, 1.U)
  step(1)
  expect(c.io.s2g_dep_queue.ready, 1.U)
  poke(c.io.s2g_dep_queue.valid, 0.U)
  step(1)

  poke(c.io.done.read, 1.U)
  step(1)
  poke(c.io.done.read, 0.U)
  step(1)
  }

  def init_compute_blocked_gemm() {
  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)
  step(1)

  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, uops_blocked_gemm_data(0))
  step(1)
  poke(c.io.gemm_queue.data, insn_blocked_gemm_uop)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)

  for (i <- 0 to 31) {
    expect(c.io.uops.address, ("h"+(BigInt("0010",16)*i).toString(16)).U)
    poke(c.io.uops.readdata, uops_blocked_gemm_data(i % 4)) // TODO: update uops data here
    step(1)
  }
  step(1)
  step(1)

  poke(c.io.gemm_queue.data, insn_blocked_gemm_acc)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  }
  

  def test_compute_min(sram_base: BigInt = 0, dram_base: BigInt = 0) {

  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 1.U)

  for (i <- 0 to 7) {
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("000",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min(i).substring(32 * 3, 32 * 3 + 32)).U)
    step(1)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("010",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min(i).substring(32 * 2, 32 * 2 + 32)).U)
    step(1)
    poke(c.io.biases.waitrequest, 1.U)
    step(1)
    poke(c.io.biases.waitrequest, 0.U)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("020",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min(i).substring(32 * 1, 32 * 1 + 32)).U)
    step(1)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("030",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min(i).substring(32 * 0, 32 * 0 + 32)).U)
    poke(c.io.biases.waitrequest, 1.U)
    step(1)
    poke(c.io.biases.waitrequest, 0.U)
    if (i == 6) {
      poke(c.io.gemm_queue.data, insn_min_out)
      poke(c.io.gemm_queue.valid, 1.U)
    }
    step(1)
  } // end of for loop

  step(6)

  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h00".U)
  expect(c.io.out_mem.writedata, "hf6c4c7efc1ecf3fcfcfcfcfceec6e9f5".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h10".U)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.writedata, "hfcfcfbfcf3defcfafcfccdfcd3c6fcfc".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h20".U)
  expect(c.io.out_mem.writedata, "hfcfcfcd8fcfcfcfcd1fcfcd0dbf8c7c7".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h30".U)
  expect(c.io.out_mem.writedata, "he2fcfcdddbfcfcfcfcd9c5f8fcf8ecfc".U)
  step(1)
  expect(c.io.out_mem.address, "h40".U)
  step(1)
  expect(c.io.out_mem.address, "h50".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.address, "h60".U)
  step(1)
  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)

  step(1)
  expect(c.io.out_mem.write, 0.U)

  expect(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

  }

  def test_compute_min_nocomp(sram_base: BigInt = 0, dram_base: BigInt = 0) {

  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 1.U)

  for (i <- 0 to 7) {
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("000",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min_nocomp(i).substring(32 * 3, 32 * 3 + 32)).U)
    step(1)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("010",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min_nocomp(i).substring(32 * 2, 32 * 2 + 32)).U)
    step(1)
    poke(c.io.biases.waitrequest, 1.U)
    step(1)
    poke(c.io.biases.waitrequest, 0.U)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("020",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min_nocomp(i).substring(32 * 1, 32 * 1 + 32)).U)
    step(1)
    expect(c.io.biases.address, ("h"+(dram_base+BigInt("030",16)+BigInt("040",16)*i).toString(16)).U)
    poke(c.io.biases.readdata, ("h"+biases_min_nocomp(i).substring(32 * 0, 32 * 0 + 32)).U)
    poke(c.io.biases.waitrequest, 1.U)
    step(1)
    poke(c.io.biases.waitrequest, 0.U)
    if (i == 6) {
      poke(c.io.gemm_queue.data, insn_min_nocomp_out)
      poke(c.io.gemm_queue.valid, 1.U)
    }
    step(1)
  } // end of for loop

  step(6)

  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h00".U)
  expect(c.io.out_mem.writedata, "h01f3e2c805df05de05e8e9db0505d6d5".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h10".U)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h10".U)
  expect(c.io.out_mem.writedata, "he6e4d9ee05e605e1d302050505d0fffb".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h20".U)
  expect(c.io.out_mem.writedata, "hddd0f6d1d80505f005e7ceed05df05e3".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h30".U)
  expect(c.io.out_mem.writedata, "h05e4d3c5050205f105e5ec0505050505".U)
  step(1)
  expect(c.io.out_mem.address, "h40".U)
  expect(c.io.out_mem.writedata, "h0505c8df04d005050505ddfbf605c9ce".U)
  step(1)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h50".U)
  expect(c.io.out_mem.writedata, "h05f5eff5e1da05050505dcecfbc4ef05".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.address, "h60".U)
  expect(c.io.out_mem.writedata, "h05050505050505d30505e1050505cac0".U)
  step(1)
  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)

  step(1)
  expect(c.io.out_mem.write, 0.U)

  step(1)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

  }

  def test_compute_max(sram_base: BigInt = 0, dram_base: BigInt = 0) {

  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 1.U)

  for (i <- 0 to 7) {

  expect(c.io.biases.address, ("h"+(dram_base+BigInt("000",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_max(i).substring(32 * 3, 32 * 3 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("010",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_max(i).substring(32 * 2, 32 * 2 + 32)).U)
  step(1)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("020",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_max(i).substring(32 * 1, 32 * 1 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("030",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_max(i).substring(32 * 0, 32 * 0 + 32)).U)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  if (i == 6) {
    poke(c.io.gemm_queue.data, insn_max_out)
    poke(c.io.gemm_queue.valid, 1.U)
  }
  step(1)

  } // end of for loop

  step(6)

  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h00".U)
  expect(c.io.out_mem.writedata, "h1a0b360402350203020202020228382e".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h10".U)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h10".U)
  expect(c.io.out_mem.writedata, "h02020202020233023a0202023b020914".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h20".U)
  expect(c.io.out_mem.writedata, "h02021c1702021b0f02212a35190d0202".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h30".U)
  expect(c.io.out_mem.writedata, "h1002070f020902022102022c021e0202".U)
  step(1)
  expect(c.io.out_mem.address, "h40".U)
  expect(c.io.out_mem.writedata, "h02022403020202380237063b02020202".U)
  step(1)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h50".U)
  expect(c.io.out_mem.writedata, "h200202070202192b0202021802020202".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.address, "h60".U)
  expect(c.io.out_mem.writedata, "h0239160202023824020d3b3802022102".U)
  step(1)
  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)

  step(1)
  expect(c.io.out_mem.write, 0.U)

  step(1)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

  }

  def test_compute_shr(sram_base: BigInt = 0, dram_base: BigInt = 0) {

  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 1.U)

  for (i <- 0 to 7) {

  expect(c.io.biases.address, ("h"+(dram_base+BigInt("000",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_shr(i).substring(32 * 3, 32 * 3 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("010",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_shr(i).substring(32 * 2, 32 * 2 + 32)).U)
  step(1)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("020",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_shr(i).substring(32 * 1, 32 * 1 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("030",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_shr(i).substring(32 * 0, 32 * 0 + 32)).U)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  if (i == 6) {
    poke(c.io.gemm_queue.data, insn_shr_out)
    poke(c.io.gemm_queue.valid, 1.U)
  }
  step(1)

  } // end of for loop

  step(6)
  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  step(1)

  for (i <- 0 to 6) {
    expect(c.io.out_mem.write, 1.U)
    expect(c.io.out_mem.address, ("h"+(BigInt("0010",16)*i).toString(16)).U)
    poke(c.io.out_mem.waitrequest, 1.U)
    step(1)
    expect(c.io.out_mem.address, ("h"+(BigInt("0010",16)*i).toString(16)).U)
    expect(c.io.out_mem.writedata, outputs_shr(i))
    poke(c.io.out_mem.waitrequest, 0.U)
    step(1)
  }

  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  step(1)
  expect(c.io.out_mem.write, 0.U)
  step(1)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)
  }


  def test_compute_blocked_gemm(sram_base: BigInt = 0, dram_base: BigInt = 0) {

  poke(c.io.gemm_queue.valid, 0.U)
  expect(c.io.biases.read, 1.U)

  for (i <- 0 to (64 * 4 - 1)) {

  expect(c.io.biases.address, ("h"+(dram_base+BigInt("000",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_blocked_gemm(i%8).substring(32 * 3, 32 * 3 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("010",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_blocked_gemm(i%8).substring(32 * 2, 32 * 2 + 32)).U)
  step(1)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("020",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_blocked_gemm(i%8).substring(32 * 1, 32 * 1 + 32)).U)
  step(1)
  expect(c.io.biases.address, ("h"+(dram_base+BigInt("030",16)+BigInt("040",16)*i).toString(16)).U)
  poke(c.io.biases.readdata, ("h"+biases_blocked_gemm(i%8).substring(32 * 0, 32 * 0 + 32)).U)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  if (i == (64 * 4 - 2)) {
    poke(c.io.gemm_queue.data, insn_blocked_gemm_out)
    poke(c.io.gemm_queue.valid, 1.U)
  }
  step(1)

  } // end of for loop

  step(6)
  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  step(1)

  expect(c.io.l2g_dep_queue.ready, 0.U)
  poke(c.io.l2g_dep_queue.data, 1.U)
  poke(c.io.l2g_dep_queue.valid, 1.U)
  step(1)
  expect(c.io.l2g_dep_queue.ready, 1.U)
  poke(c.io.l2g_dep_queue.valid, 0.U)
  step(2)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(4)

  for (i <- 0 to 31) {
    val j = i % 16
    expect(c.io.out_mem.write, 1.U)
    expect(c.io.out_mem.address, ("h"+(BigInt("0040",16)*j).toString(16)).U)
    poke(c.io.out_mem.waitrequest, 1.U)
    step(1)
    expect(c.io.out_mem.address, ("h"+(BigInt("0040",16)*j).toString(16)).U)
    // expect(c.io.out_mem.writedata, outputs_blocked_gemm(i))
    poke(c.io.out_mem.waitrequest, 0.U)
    step(1)
  }
  
  // expect(c.io.out_mem.address, "h70".U)
  // poke(c.io.out_mem.waitrequest, 0.U)
  // step(1)
  // step(1)
  // expect(c.io.out_mem.write, 0.U)
  // step(1)
  // expect(c.io.g2s_dep_queue.valid, 1.U)
  // poke(c.io.g2s_dep_queue.ready, 1.U)
  // step(1)
  // poke(c.io.g2s_dep_queue.ready, 0.U)
  // step(1)
  // step(1)
  // step(1)
  // step(1)
  }
  
  init_compute_min()
  test_compute_min()

  init_compute_min_rebase()
  test_compute_min(0, BigInt("200", 16))

  test_compute_finish()

  init_compute_min_nocomp()
  test_compute_min_nocomp()

  init_compute_max()
  test_compute_max()

  init_compute_shr()
  test_compute_shr()

  init_compute_blocked_gemm()
  test_compute_blocked_gemm()
}

class ComputeTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "Compute"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new Compute(), backend)((c) => new ComputeTests(c)) should be (true)
    }
  }
}

