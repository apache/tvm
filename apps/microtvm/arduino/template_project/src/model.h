#ifndef IMPLEMENTATION_H_
#define IMPLEMENTATION_H_

#define WORKSPACE_SIZE $workspace_size_bytes

#ifdef __cplusplus
extern "C" {
#endif

void TVMInitialize();

// TODO template these void* values once MLF format has input and output data
void TVMExecute(void* input_data, void* output_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IMPLEMENTATION_H_
