/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

  .text
  .globl  lwp_handler
  .falign
  .type  lwp_handler,@function
lwp_handler:
  { allocframe(#24)
    memd(r29+#-16) = r5:4
  }
  {
    memd(r29+#8) = r3:2
    memd(r29+#0) = r1:0
    r2 = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
  }
  {
    r5 = memw(r2+##__lwp_counter@GOT)
    r3 = memw(r2+##__lwp_buffer_count@GOT)
  }
  {
    r5 = memw(r5+#0)
    r3 = memw(r3+#0)
  }
  {
    r4 = memw(r5+r0<<#2)
    r1 = memw(r2+##__lwp_buffer_size@GOT)
  }
  {
    r4 = add(r4,#1)
    memw(r5+r0<<#2) = r4.new
    r1 = memw(r1+#0)
  }
  {
    p0 = cmp.gtu(r4,#100)
    if (p0.new) jump:nt .LBB0_3
    r5 = memw(r2+##__lwp_buffer_ptr@GOT)
  }
  {
    r5 = memw(r5+#0)
    r2 = memw(r2+##__lwp_buffer_count@GOT)
  }
  {
    r4 = add(r3,#4)
    if (!cmp.gtu(r1,r4.new)) jump:t .LBB0_3
  }
  {
    r5 = addasl(r5,r3,#2)
    memw(r2+#0) = r4
  }
  {
    memw(r5+#0) = r31
    r1:0 = C15:14
    memw(r5+#4) = r0 // id
  }
  {
    memw(r5+#12) = r1 // pcyclehi
    memw(r5+#8) = r0 // pcyclelo
  }
  .falign
.LBB0_3:
  {
    r5:4 = memd(r29+#16)
    r3:2 = memd(r29+#8)
  }
  {
    r1:0 = memd(r29+#0)
    dealloc_return
  }
.Lfunc_end0:
  .size  lwp_handler, .Lfunc_end0-lwp_handler
