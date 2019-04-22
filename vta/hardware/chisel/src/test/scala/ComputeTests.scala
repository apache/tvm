// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class ComputeTests(c: Compute)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

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

  // shr
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
  expect(c.io.g2s_dep_queue.valid, 1.U)
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
  expect(c.io.g2s_dep_queue.valid, 1.U)
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
  expect(c.io.g2s_dep_queue.valid, 1.U)
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
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h00".U)
  expect(c.io.out_mem.writedata, "hf5f0f5faf7f6f1040dfe000800b0ff1e".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h10".U)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h10".U)
  expect(c.io.out_mem.writedata, "h06f0f9f1fd04f10cff0cf1f104fa07f1".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h20".U)
  expect(c.io.out_mem.writedata, "h0ef5f1f601fafc06f0f5f5f70f0df1f2".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h30".U)
  expect(c.io.out_mem.writedata, "h030c04fe0c080b0cfd0afbf8f602fcf3".U)
  step(1)
  expect(c.io.out_mem.address, "h40".U)
  expect(c.io.out_mem.writedata, "h080e080ff50c0a06f2f00001fa00fa07".U)
  step(1)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h50".U)
  expect(c.io.out_mem.writedata, "hf10803fd05f6f20f0cfaf40f0bfe0f0f".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.address, "h60".U)
  expect(c.io.out_mem.writedata, "h03f50c0ff4060d0ef60bf50f08040dfb".U)
  step(1)
  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)

  step(1)
  expect(c.io.out_mem.write, 0.U)

  step(1)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.ready, 1.U)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

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

