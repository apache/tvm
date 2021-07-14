#ifndef Model_h
#define Model_h

#include "standalone_crt/include/tvm/runtime/crt/graph_executor.h"


class Model
{

  public:
    Model();
    void inference(void *input_data, void *output_data);
    int infer_category(void *input_data);

  private:
    TVMGraphExecutor* graph_runtime;
};

#endif

