#include "test_lib.h"
#include <inttypes.h>

extern "C" void printAcc(acc_T** imm, int size) {
	for(int i = 0; i < size; i++) {
		// acc_T* temp = *(imm + i*size);
		// printf("address %d is %p\n", i, temp);
		for(int j = 0; j < size; j++) {
			printf("%" PRId32 ", ", imm[i][j]);
		}
		printf("\n");
	}
}

extern "C" void transfer(void* target, void* data, size_t size) {
	memcpy(target, data, size);
}

extern "C" void concatUop(VTAUop* src, VTAUop* item, size_t size1, size_t size2) {
	memcpy(src + size1, item, size2 * sizeof(VTAUop));
}
