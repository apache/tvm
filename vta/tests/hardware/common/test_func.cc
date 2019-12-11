#include "test_lib.h"

extern "C" void transfer(void* target, void* data, size_t size) {
	memcpy(target, data, size);
}

extern "C" void concatUop(VTAUop* src, VTAUop* item, size_t size1, size_t size2) {
	memcpy(src + size1, item, size2 * sizeof(VTAUop));
}
