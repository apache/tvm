#ifndef Parameters_h
#define Parameters_h

#include "standalone_crt/include/dlpack/dlpack.h"

// Some Arduinos (like the Spresense) have multiple CPUs,
// so this could be expaneded at some point
static const DLDevice HARDWARE_DEVICE = {kDLCPU, 0};

static const int INPUT_DATA_DIMENSION = $input_data_dimension;
static const int64_t INPUT_DATA_SHAPE[] = $input_data_shape;
static const DLDataType INPUT_DATA_TYPE = $input_data_type;
static const char* INPUT_LAYER = $input_layer_name;

static const int OUTPUT_DATA_DIMENSION = $output_data_dimension;
static const int64_t OUTPUT_DATA_SHAPE[] = $output_data_shape;
static const DLDataType OUTPUT_DATA_TYPE = $output_data_type;

#endif
