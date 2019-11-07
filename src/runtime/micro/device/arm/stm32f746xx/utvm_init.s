.syntax unified
.cpu cortex-m7
.fpu softvfp
.thumb

.section .text.UTVMInit
.type UTVMInit, %function
UTVMInit:
  /* enable fpu */
  ldr r0, =0xE000ED88
  ldr r1, [r0]
  ldr r2, =0xF00000
  orr r1, r2
  str r1, [r0]
  dsb
  isb
  /* set stack pointer */
  ldr sp, =_utvm_stack_pointer_init
  bl UTVMMain
.size UTVMInit, .-UTVMInit
