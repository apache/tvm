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

/*
Lightweight profiling handler to record processor cycles in a buffer
(pointed by __lwp_buffer_ptr) for a given invocation of the handler. To keep the
buffer size within a resonable limit, we only recond data for the first 100
invocation of the handler for a given loop or function ID (passed in R0 register).
The buffer size wouldn't be a concern if the loops with only siblings are getting
profiled. However, since the instrumentation provides several different profiling
options, this approach ensures that they all function as expexted. We use second
buffer (pointed by __lwp_counter) to keep count of the calls made to lwp_handler
function for each function/loop.

Brief explanation of all the global variables used:
1) __lwp_counter : Pointer to the buffer that keeps count of the number of times handler
 is called for a given ID. To reduce the complexity of the handler, __lwp_counter is
 indexed using the ID itself.
2) __lwp_buffer_ptr : Pointer to the buffer that records loop/function ID, processor cycles
 and return addresss of the handler. Return address is used to reconstruct the call graph
 (loop-nest) to make it easier to analyze the profiling data.
3) __lwp_buffer_size : Size of the buffer
4) __lwp_buffer_count : Offset into main lwp buffer where data for the current handler
invocation needs to be written.

NOTE: The handler function saves and restores R0-R5 registers which are caller saved registers
on Hexagon and should be handled at the callsite. However, to reduce the codegen impact
of the handler calls on the caller functions, we decided to move this part into the
handler itself.

*/
  .text
  .globl  lwp_handler
  .falign
  .type  lwp_handler,@function
lwp_handler:
  {
    allocframe(#32)                            // Allocate 32 bytes on the stack to save R0-R5 registers (6*4bytes) and P0-P3 (4*1byte) + 4 unused bytes as the stack has to be 8-bytes aligned
    memd(r29+#-16) = r5:4                      // Save R5,R4
    r5 = p3:0                                  // We will save P3:0 but we need an intermediate usual register (R5) that has already been saved
  }
  {
    memd(r29+#16) = r3:2                       // Save R3,R2
    memd(r29+#8) = r1:0                        // Save R1, R0
  }
  {
    memw(r29+#0) = r5                          // Save P3:0 (via R5)
    r2 = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL) // Get GOT address
  }
  {
    r5 = memw(r2+##__lwp_counter@GOT)         // Get address of the pointer to __lwp_counter
    r3 = memw(r2+##__lwp_buffer_count@GOT)    // Get the address of __lwp_buffer_count
  }
  {
    r5 = memw(r5+#0)                          // Get the address of __lwp_counter (address of the main lwp buffer)
    r3 = memw(r3+#0)                          // Get the __lwp_buffer_count value (offset into the main buffer)
  }
  {
    r4 = memw(r5+r0<<#2)                      // Get the handler invocation count for the ID (passed in R0)
    r1 = memw(r2+##__lwp_buffer_size@GOT)     // Get the address of __lwp_buffer_size
  }
  {
    r4 = add(r4,#1)                           // Increment count
    memw(r5+r0<<#2) = r4.new                  // Update count in __lwp_counter for a given ID
    r1 = memw(r1+#0)                          // Get the buffer size
  }
  {
    p0 = cmp.gtu(r4,#100)                     // Exit if count for a given ID is greater than 100
    if (p0.new) jump:nt .LBB0_3
    r5 = memw(r2+##__lwp_buffer_ptr@GOT)      // Get address of the pointer to __lwp_buffer_ptr
  }
  {
    r5 = memw(r5+#0)                          // Get address of __lwp_buffer_ptr
    r2 = memw(r2+##__lwp_buffer_count@GOT)    // Get address of __lwp_buffer_count
  }
  {
    r4 = add(r3,#4)                           // Increment the offset by 4 since 4 int32 values are stored for each invocation
    if (!cmp.gtu(r1,r4.new)) jump:t .LBB0_3   // Exit if the main lwp buffer has run out of space
  }
  {
    r5 = addasl(r5,r3,#2)                     // Get the address where the data needs to be recorded
    memw(r2+#0) = r4                          // Save next offset into __lwp_buffer_count
  }
  {
    memw(r5+#0) = r31                         // Save return address of this function
    r1:0 = C15:14                             // Control registers that keep processor cycle count (64-bits)
    memw(r5+#4) = r0                          // Save loop/function ID
  }
  {
    memw(r5+#12) = r1                         // Save upper 32 bits
    memw(r5+#8) = r0                          // Save lower 32 bits
  }
  .falign
.LBB0_3:                                      // Restore the registers from the stack
  {
    r1 = memw(r29+#0)                         // We will restore P3:0 but need an intermediate usual register (R1) that hasn't already been restored
    r5:4 = memd(r29+#24)                      // Restore R5:4
  }
  {
    r3:2 = memd(r29+#16)                      // Restore R3:2
    p3:0 = r1                                 // Restore P3:0 (via R1, not yet restored)
  }
  {
    r1:0 = memd(r29+#8)                       // Restore R1:0
    dealloc_return                            // Deallocate the stack and return
  }
.Lfunc_end0:
  .size  lwp_handler, .Lfunc_end0-lwp_handler
