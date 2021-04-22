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
#ifndef TVM_RUNTIME_SUBGRAPH_SUBGRAPH_STRUCT_H_
#define TVM_RUNTIME_SUBGRAPH_SUBGRAPH_STRUCT_H_
#include <assert.h>
#include <sched.h>
#include <string.h>
#include <sys/syscall.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <unistd.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define SLOT slot_t<>
#define SUB_Q_SIZE 1024
using namespace tvm::runtime;
using namespace std;
// thread control struction, for single consumer single producer mode
class TControl {
 private:
  condition_variable cond;
  volatile bool bWait = false;
  mutex m;

 public:
  volatile bool bExit = false;
  bool wait(bool bPollSuc) {
    if (bPollSuc) {
      return true;
    }

    unique_lock<mutex> lock(m);
    cond.wait(lock, [&] { return this->bWait; });
    bWait = false;

    return !bExit;
  }

  void notify(void) {
    bWait = true;
    cond.notify_one();
  }

  void exit_notify(thread* t) {
    /*
     * set bExit first then notify
     */
    bExit = true;
    notify();
    if (t->joinable()) {
      t->join();
    }
  }
};

class subgraphData {
 private:
  void ResetDataList(size_t num) {
    if (max_num < num) {
      for (size_t i = 0; i < max_num; i++) {
        TVMArrayFree(dataList[i]);
      }

      if (dataList) {
        free(dataList);
      }

      dataList = reinterpret_cast<DLTensor**>(calloc(num, sizeof(DLTensor*)));
      max_num = num;
    }
    return;
  }

  DLTensor* CreateCopyFrom(const DLTensor* from, DLTensor** to, int device_type, int device_id) {
    size_t fromLen = tvm::runtime::GetDataSize(*from);
    size_t toLen = *to ? tvm::runtime::GetDataSize(*(*to)) : 0;

    if (fromLen != toLen) {
      if (*to) {
        TVMArrayFree(*to);
        *to = nullptr;
      }
      TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits, from->dtype.lanes,
                    device_type, device_id, to);
    }
    TVMArrayCopyFromTo(const_cast<DLTensor*>(from), *to, nullptr);
    return *to;
  }

 public:
  void Copy(const Array<NDArray>& dlArray, int device_type, int device_id) {
    num = dlArray.size();
    ResetDataList(num);

    for (size_t i = 0; i < num; i++) {
      CreateCopyFrom(const_cast<const DLTensor*>(dlArray[i].operator->()), &dataList[i],
                     device_type, device_id);
    }
    return;
  }

  void Copy(const DLTensor* dlTensor, int device_type, int device_id) {
    num = 1;
    ResetDataList(num);
    CreateCopyFrom(dlTensor, &dataList[0], device_type, device_id);
    return;
  }

  void Copy(const vector<const DLTensor*>& dlTensors, int device_type, int device_id) {
    num = dlTensors.size();
    ResetDataList(num);

    for (size_t i = 0; i < num; i++) {
      CreateCopyFrom(dlTensors[i], &dataList[i], device_type, device_id);
    }
    return;
  }

  void Copy(DLTensor** dlTensors, size_t dlNum, int device_type, int device_id) {
    num = dlNum;
    ResetDataList(num);

    for (size_t i = 0; i < num; i++) {
      auto dlTensor = const_cast<DLTensor*>(dlTensors[i]);
      CreateCopyFrom(dlTensor, &dataList[i], device_type, device_id);
    }
    return;
  }
  size_t num;
  size_t max_num;
  DLTensor** dataList;
  TControl controlData;
  subgraphData(void) : num(0), max_num(0), dataList(nullptr) {}
};

template <int device_type = kDLCPU, int device_id = 0>
class slot_t {
 public:
  bool bExit = false;
  subgraphData data;
  slot_t(void) {}

  // overwrite operator = to handle "(slot) s = (OutputData) d;"
  slot_t<device_type, device_id>& operator=(const DLTensor* dlTensor) {
    data.Copy(dlTensor, device_type, device_id);
    return *this;
  }

  slot_t<device_type, device_id>& operator=(const vector<const DLTensor*> dlTensors) {
    data.Copy(dlTensors, device_type, device_id);
    return *this;
  }

  slot_t<device_type, device_id>& operator=(const Array<NDArray> dlTensors) {
    data.Copy(dlTensors, device_type, device_id);
    return *this;
  }

  slot_t<device_type, device_id>& operator=(const slot_t<device_type, device_id>& slot) {
    data.Copy(slot.data.dataList, slot.data.num, device_type, device_id);
    return *this;
  }
};

template <int device_type = kDLCPU, int device_id = 0>
class subgraphOutputData {
 public:
  explicit subgraphOutputData(vector<NDArray>* datas) : datas_(datas) { ; }
  subgraphOutputData& operator=(const slot_t<device_type, device_id>& slot) {
    assert(datas_->size() >= slot.data.num);
    for (size_t i = 0; i < slot.data.num; i++) {
      auto dlTensor = slot.data.dataList[i];
      (*datas_)[i].CopyFrom(dlTensor);
    }
    return *this;
  }

 private:
  vector<NDArray>* datas_;
};

template <typename SLOT_TYPE = SLOT, int QLEN = 1024>
class squeue {
 public:
  size_t len;
  volatile size_t head;
  volatile size_t tail;
  SLOT_TYPE q[QLEN];
  squeue(void) : len(QLEN), head(0), tail(0) {}
};
typedef squeue<SLOT> QUEUE;

class RuntimeFunction {
 public:
  DLTensor* dlLocal = nullptr;
  Module module_;
  tvm::runtime::PackedFunc get_num_output;
  tvm::runtime::PackedFunc get_num_inputs;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc get_input;
  tvm::runtime::PackedFunc run;
  explicit RuntimeFunction(const Module& m) {
    module_ = m;
    get_num_output = module_.GetFunction("get_num_outputs");
    get_num_inputs = module_.GetFunction("get_num_inputs");
    set_input = module_.GetFunction("set_input");
    get_output = module_.GetFunction("get_output");
    get_input = module_.GetFunction("get_input");
    run = module_.GetFunction("run");
  }
  ~RuntimeFunction() {
    if (dlLocal) {
      TVMArrayFree(dlLocal);
      dlLocal = nullptr;
    }
  }

  DLTensor* CreateFromDLTensor(const DLTensor* from) {
    DLTensor* ret = NULL;
    TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits, from->dtype.lanes,
                  kDLCPU, 0, &ret);
    return ret;
  }

  int NumOutputs() const { return get_num_output(); }
  int NumInputs() const { return get_num_inputs(); }

  /*
     when doing subgraph pipeline, the from data and to
     data may comming from different device, for example
     one from GPU another from VTA, here we need first
     copy it into cpu type memory from GPU then copy the
     cpu type memory into VTA, because current NDArray
     copy not support cross device memory copy.
     */
  void CopyFromTo(DLTensor* from, DLTensor* to) {
    if (!(from->device.device_type == to->device.device_type ||
          from->device.device_type == kDLCPU || to->device.device_type == kDLCPU ||
          from->device.device_type == kDLCPUPinned || to->device.device_type == kDLCPUPinned)) {
      if (dlLocal == nullptr) {
        dlLocal = CreateFromDLTensor(from);
      }
      TVMArrayCopyFromTo(from, dlLocal, nullptr);
      from = dlLocal;
    }

    TVMArrayCopyFromTo(from, to, nullptr);
  }

  void SetInput(int index, DLTensor* data_in) {
    /*
       Here we can not use 'GetInput' of this class to replace
       'get_input' although it just be one more level wrap for
       'get_input', doing one more level wrap would
       cause a NDArray copy and deconstruction after GetInput call,
       when such NDArray comming from a RPC value, the deconstruction may
       cause the remote data get free. then following operation for
       such NDArray which linked a corrupt data would cause crash.
       */
    NDArray input = get_input(index);
    DLTensor* dlInput = const_cast<DLTensor*>(input.operator->());
    CopyFromTo(data_in, dlInput);
  }

  void SetInput(const std::string& name, DLTensor* data_in) {
    NDArray input = get_input(name);
    DLTensor* dlInput = const_cast<DLTensor*>(input.operator->());
    CopyFromTo(data_in, dlInput);
  }

  NDArray GetInput(const std::string& name) { return get_input(name); }

  NDArray GetOutput(int index) const { return get_output(index); }

  NDArray GetInput(int index) const { return get_input(index); }

  void Run() { run(); }
};

class RuntimeData {
 private:
  shared_ptr<RuntimeFunction> runtimePtr;
  template <typename type>
  void ImportData(type dlTensors, size_t inputsLen) {
    assert(runtimePtr->NumInputs() >= inputsLen);
    for (size_t i = 0; i < inputsLen; i++) {
      /*
       * Use SetInput which have logic to handle
       * cross device memory copy to set input data.
       */
      runtimePtr->SetInput(i, dlTensors[i]);
    }
    return;
  }

 public:
  void Init(shared_ptr<RuntimeFunction> runtime) { runtimePtr = runtime; }

  RuntimeData& operator=(const SLOT& slot) {
    ImportData<DLTensor**>(slot.data.dataList, slot.data.num);
    return *this;
  }

  RuntimeData& operator=(vector<DLTensor*> dlTensors) {
    ImportData<vector<DLTensor*>>(dlTensors, dlTensors.size());
    return *this;
  }
};

class RuntimeItem {
 public:
  shared_ptr<RuntimeItem> prev = nullptr;
  shared_ptr<RuntimeItem> next = nullptr;

  int inputsNum;
  RuntimeData rData;
  TControl control;
  QUEUE* queue = nullptr;
  thread t;
  shared_ptr<RuntimeFunction> runtimePtr = nullptr;
  RuntimeItem(Module mod, QUEUE* inputQueue) {
    if (runtimePtr == nullptr) {
      runtimePtr = make_shared<RuntimeFunction>(mod);
      inputsNum = runtimePtr->NumOutputs();
      rData.Init(runtimePtr);
    }

    if (!queue) {
      queue = inputQueue;
    }
  }

  RuntimeItem(void) {}

  void Run(void) { runtimePtr->Run(); }

  bool waitPipeLineData(bool bPollSuc) {
    /*
       wait input data ready.
       */
    return control.wait(bPollSuc);
  }

  void notifyDataReadyToNext(void) {
    if (next) {
      next->control.notify();
    }
  }

  void notifyNextExit(void) {
    if (next) {
      next->control.exit_notify(&next->t);
    }
  }

  /*
   * Here we need to use a container to storage NDArray that from
   * GetOutput, if just copy the data but not storage NDArray, the
   * memory of data may get freed, especially for RPC device data,
   */
  Array<NDArray> GetOutput(void) {
    Array<NDArray> outputs;
    size_t outputsNum = runtimePtr->NumOutputs();
    for (size_t i = 0; i < outputsNum; i++) {
      auto output = runtimePtr->GetOutput(i);
      outputs.push_back(output);
    }
    return outputs;
  }
};

#endif  //  TVM_RUNTIME_SUBGRAPH_SUBGRAPH_STRUCT_H_
