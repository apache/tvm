//#include "stm32f7xx.h"

//extern char _stext;
void _UTVMInit(void)
{
//  /* FPU settings ------------------------------------------------------------*/
//  #if (__FPU_PRESENT == 1) && (__FPU_USED == 1)
//    SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));  /* set CP10 and CP11 Full Access */
//  #endif
//  /* Reset the RCC clock configuration to the default reset state ------------*/
//  /* Set HSION bit */
//  RCC->CR |= (uint32_t)0x00000001;
//
//  /* Reset CFGR register */
//  RCC->CFGR = 0x00000000;
//
//  /* Reset HSEON, CSSON and PLLON bits */
//  RCC->CR &= (uint32_t)0xFEF6FFFF;
//
//  /* Reset PLLCFGR register */
//  RCC->PLLCFGR = 0x24003010;
//
//  /* Reset HSEBYP bit */
//  RCC->CR &= (uint32_t)0xFFFBFFFF;
//
//  /* Disable all interrupts */
//  RCC->CIR = 0x00000000;
//
//  /* Configure the Vector Table location add offset address ------------------*/
//#ifdef VECT_TAB_SRAM
//  SCB->VTOR = RAMDTCM_BASE | VECT_TAB_OFFSET; /* Vector Table Relocation in Internal SRAM */
//#else
//  SCB->VTOR = FLASH_BASE | VECT_TAB_OFFSET; /* Vector Table Relocation in Internal FLASH */
//#endif
  //SCB->VTOR = &_stext; /* Vector Table Relocation in Internal SRAM */
}
