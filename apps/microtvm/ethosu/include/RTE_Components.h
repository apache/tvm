
#ifndef RTE_COMPONENTS_H
#define RTE_COMPONENTS_H


#ifdef M55_HP
#define CMSIS_device_header "M55_HP.h"
#elif defined M55_HE
#define CMSIS_device_header "M55_HE.h"
#else
#define CMSIS_device_header "ARMCM55.h"
#endif

#include CMSIS_device_header


#define RTE_Drivers_GPIO        1
#define RTE_Drivers_PINCONF     1
#define RTE_UART4               1
#define RTE_UART0               1

#define RTE_Drivers_CAMERA0     0
#define RTE_Drivers_I3C0        0
#define RTE_Drivers_SAI         0

#endif /* RTE_COMPONENTS_H */
