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
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
#include <assert.h>
#include <sched.h>
#include <string.h>
#include <sys/syscall.h>
//#include <tvm/runtime/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <unistd.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#define SLOT slot_t<>
#define SUB_Q_SIZE 1024

using namespace tvm::runtime;
using namespace std;
typedef unordered_map<int, unordered_map<int, int>> RUNTIME_PIPELINE_OUTPUT_CONF;
typedef unordered_map<int, unordered_map<int, string>> RUNTIME_PIPELINE_OUTPUT_CONF_STR;
/* thread control struction, for single consumer single producer mode.
 */
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
    /* set bExit first then notify
     */
    bExit = true;
    notify();
    if (t->joinable()) {
      t->join();
    }
  }
};

#define DEPENDENT_MAX 32
#define TYP_MAX(type) (1 << size_of(type) - 1)
typedef uint8_t DEP_INDX_TYPE;
class Dependent {
 private:
  /* index 0  represent output is final output or not.*/
  uint8_t bFinal = false;
  /* how many dependent*/
  uint8_t depNum = 0;
  /* dependent input index number.*/
  union {
    DEP_INDX_TYPE dependent[DEPENDENT_MAX] = {0};
    DEP_INDX_TYPE outputIndx;
  };

 public:
  void SetDepModInputIndx(const int modIndx, const uint8_t inputIndx) {
    assert(modIndx <= DEPENDENT_MAX);
    assert(inputIndx <= TYP_MAX(DEP_INDX_TYPE));
    if (modIndx == 0) {
      bFinal = true;
      outputIndx = inputIndx - 1;
    } else {
      dependent[modIndx - 1] = inputIndx;
    }
    depNum++;
  }

  int GetOutputIndx(void) { return outputIndx; }

  int GetDepModInputIndx(const int modIndx) { return dependent[modIndx - 1]; }

  void RemoveDependentRef(const int modIndx) {
    dependent[modIndx - 1] = 0;
    depNum--;
  }

  /*
   * check if the output need get forward to next runtime.
   */
  bool NeedForward() { return (bFinal || depNum > 0); }
};

class TensorData {
 public:
  DLTensor* data = nullptr;

  DLTensor* CreateCopyFrom(const DLTensor* from, int device_type, int device_id) {
    size_t fromLen = tvm::runtime::GetDataSize(*from);
    size_t toLen = data ? tvm::runtime::GetDataSize(*data) : 0;

    if (fromLen != toLen) {
      if (data) {
        TVMArrayFree(data);
        data = nullptr;
      }
      TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits, from->dtype.lanes,
                    device_type, device_id, &data);
    }
    TVMArrayCopyFromTo(const_cast<DLTensor*>(from), data, nullptr);
    return data;
  }
  ~TensorData() {
    if (data) {
      TVMArrayFree(data);
      data = nullptr;
    }
  }
};

class InputData {
 public:
  Dependent dependent;
  TensorData dlData;

  DLTensor* CreateCopyFrom(const DLTensor* from, int device_type, int device_id) {
    dlData.CreateCopyFrom(from, device_type, device_id);
    return dlData.data;
  }
};

class OutputData {
 public:
  OutputData(const NDArray& data, const size_t Indx,
             RUNTIME_PIPELINE_OUTPUT_CONF runtime_pipeline_output_conf) {
    assert(runtime_pipeline_output_conf.size() < DEPENDENT_MAX);
    /* use data_ to keep the NDArray data reference, to avoid memory
     * used by DLTensor get freed.
     */
    data_ = data;
    dltensor = const_cast<DLTensor*>(data_.operator->());
    outputIndx = Indx;
    for (auto conf : runtime_pipeline_output_conf[outputIndx]) {
      dependent.SetDepModInputIndx(conf.first, conf.second);
    }
  }

  OutputData(const int modIndx, const int inputIndx, const DLTensor* data) {
    dltensor = const_cast<DLTensor*>(data);
    dependent.SetDepModInputIndx(modIndx, inputIndx);
  }

  explicit OutputData(const InputData* pdata) {
    dependent = pdata->dependent;
    /* caller need make sure pdata->dlData.data is avaialble.
     */
    dltensor = pdata->dlData.data;
  }

  OutputData& operator=(const InputData* pdata) {
    dependent = pdata->dependent;
    /* caller need make sure pdata->dlData.data is avaialble.
     */
    dltensor = pdata->dlData.data;
    return *this;
  }

  int runtimeIdx;
  /* reserved, for debug purpose
   */
  int outputIndx;
  /* index 0  represent output is final output or not.
   * index offset is dependent mod index,
   * value is dependent mode input index
   */
  Dependent dependent;
  DLTensor* dltensor;

 private:
  NDArray data_;
};

class PipelineData {
 private:
  void FreeData() {
    for (size_t i = 0; i < max_num; i++) {
      delete inputList[i];
    }

    if (inputList) {
      free(inputList);
    }
  }

  void ResetDataList(size_t num) {
    if (max_num < num) {
      FreeData();
      inputList = reinterpret_cast<InputData**>(calloc(num, sizeof(InputData)));
      max_num = num;
    }
    return;
  }

  InputData* CreateCopyFrom(const DLTensor* fromData, const Dependent& fromDep, InputData** to,
                            int device_type, int device_id) {
    if (!*to) {
      *to = new InputData;
    }

    (*to)->CreateCopyFrom(fromData, device_type, device_id);
    (*to)->dependent = fromDep;
    return *to;
  }

 public:
  void ExportAppendData(vector<shared_ptr<OutputData>>* outputs) {
    for (size_t i = 0; i < num; i++) {
      shared_ptr<OutputData> var = make_shared<OutputData>(inputList[i]);
      outputs->push_back(var);
    }
    return;
  }

  void Copy(const vector<InputData*>& dlInput, int device_type, int device_id) {
    num = dlInput.size();
    ResetDataList(num);

    for (size_t i = 0; i < num; i++) {
      CreateCopyFrom(dlInput[i]->dlData.data, dlInput[i]->dependent, &inputList[i], device_type,
                     device_id);
    }
    return;
  }

  void Copy(const vector<shared_ptr<OutputData>>* dlOutput, int device_type, int device_id) {
    num = dlOutput->size();
    ResetDataList(num);

    for (size_t i = 0; i < num; i++) {
      CreateCopyFrom(dlOutput->at(i)->dltensor, dlOutput->at(i)->dependent, &inputList[i],
                     device_type, device_id);
    }
    return;
  }

  size_t num;
  size_t max_num;
  InputData** inputList;

  TControl controlData;
  PipelineData(void) : num(0), max_num(0), inputList(nullptr) {}
  ~PipelineData(void) { FreeData(); }
};

template <int device_type = kDLCPU, int device_id = 0>
class slot_t {
 public:
  bool bExit = false;
  PipelineData data;
  slot_t(void) {}

  slot_t<device_type, device_id>& operator=(const vector<shared_ptr<OutputData>>* outputData) {
    data.Copy(outputData, device_type, device_id);
    return *this;
  }
};

template <int device_type = kDLCPU, int device_id = 0>
class pipelineOutputData {
 public:
  explicit pipelineOutputData(vector<NDArray>* datas) : datas_(datas) { ; }
  pipelineOutputData& operator=(const slot_t<device_type, device_id>& slot) {
    assert(datas_->size() >= slot.data.num);
    unordered_map<int, DLTensor*> dataMap;
    /* output may not ordered by index in slot, use a map to index them.
     */
    for (size_t i = 0; i < slot.data.num; i++) {
      auto dlTensor = slot.data.inputList[i]->dlData.data;
      int outputIndx = slot.data.inputList[i]->dependent.GetOutputIndx();
      assert(outputIndx < slot.data.num);
      dataMap[outputIndx] = dlTensor;
    }

    for (size_t i = 0; i < dataMap.size(); i++) {
      auto dlTensor = dataMap[i];
      /* alloc NDArray if there is no NDArray Allocated in vector
       */
      if (datas_->size() < i + 1) {
        /* allocated NDArray
         */
        vector<int64_t> shape;
        for (int i = 0; i < dlTensor->ndim; i++) {
          shape.push_back(dlTensor->shape[i]);
        }
        auto ndarray = NDArray::Empty(shape, dlTensor->dtype, dlTensor->device);
        ndarray.CreateView(shape, dlTensor->dtype);

        /* push into NDArray vector
         */
        datas_->push_back(ndarray);
      }
      datas_->at(i).CopyFrom(dlTensor);
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
  tvm::runtime::PackedFunc get_input_index;
  tvm::runtime::PackedFunc run;
  explicit RuntimeFunction(const Module& m) {
    module_ = m;
    get_num_output = module_.GetFunction("get_num_outputs");
    get_num_inputs = module_.GetFunction("get_num_inputs");
    set_input = module_.GetFunction("set_input");
    get_output = module_.GetFunction("get_output");
    get_input = module_.GetFunction("get_input");
    get_input_index = module_.GetFunction("get_input_index");
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
     when doing pipeline, the from data and to
     data may comming from different device, for example
     one from GPU another from VTA, here we need first
     copy it into cpu type memory from GPU then copy the
     cpu type memory into VTA, because current NDArray
     copy not support cross device memory copy.
     */
  void CopyFromTo(DLTensor* from, DLTensor* to) {
    if (!(from->device.device_type == to->device.device_type ||
          from->device.device_type == kDLCPU || to->device.device_type == kDLCPU)) {
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

  int GetInputIndex(const std::string& name) { return get_input_index(name); }

  void Run() { run(); }
};

class RuntimeData {
 private:
  shared_ptr<RuntimeFunction> runtimePtr;
  int runtimeIndx;
  /* Storage these data that need to get forwarding to
   * next runtime.
   */
  PipelineData forwardData;

  void ImportData(vector<DLTensor*> dlTensors, size_t inputsLen) {
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

  void ImportPipelineData(InputData** data, size_t inputsLen) {
    assert(runtimePtr->NumInputs() >= inputsLen);
    vector<InputData*> forwardDatas;
    for (size_t i = 0; i < inputsLen; i++) {
      /*
       * Use SetInput which have logic to handle
       * cross device memory copy to set input data.
       */
      int inputIndx = data[i]->dependent.GetDepModInputIndx(runtimeIndx);
      if (inputIndx > 0) {
        runtimePtr->SetInput(inputIndx - 1, data[i]->dlData.data);
        /* data getused remove dependent reference for current runtime
         */
        data[i]->dependent.RemoveDependentRef(runtimeIndx);
      }

      /* save these data that need forwarding to next runtime.
       */
      if (data[i]->dependent.NeedForward()) {
        forwardDatas.push_back(data[i]);
      }
    }

    forwardData.Copy(forwardDatas, kDLCPU, 0);
    return;
  }

 public:
  void ExportAppendData(vector<shared_ptr<OutputData>>* outputs) {
    forwardData.ExportAppendData(outputs);
    return;
  }

  void Init(shared_ptr<RuntimeFunction> runtime, int Indx) {
    runtimeIndx = Indx;
    runtimePtr = runtime;
  }

  RuntimeData& operator=(const SLOT& slot) {
    ImportPipelineData(slot.data.inputList, slot.data.num);

    return *this;
  }
};

class RuntimeItem {
 public:
  shared_ptr<RuntimeItem> prev = nullptr;
  shared_ptr<RuntimeItem> next = nullptr;

  RUNTIME_PIPELINE_OUTPUT_CONF runtime_pipeline_output_conf;
  int runtimeIndx;
  int inputsNum;
  RuntimeData rData;
  TControl control;
  QUEUE* queue = nullptr;
  thread t;
  shared_ptr<RuntimeFunction> runtimePtr = nullptr;
  RuntimeItem(Module mod, QUEUE* inputQueue, RUNTIME_PIPELINE_OUTPUT_CONF* pconfig, int indx) {
    if (runtimePtr == nullptr) {
      runtimePtr = make_shared<RuntimeFunction>(mod);
      inputsNum = runtimePtr->NumOutputs();
      runtimeIndx = indx;
      rData.Init(runtimePtr, runtimeIndx);
    }

    if (!queue) {
      queue = inputQueue;
    }
    runtime_pipeline_output_conf = *pconfig;
    runtimeIndx = indx;
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

  void GetOutput(vector<shared_ptr<OutputData>>* outputs) {
    size_t outputsNum = runtimePtr->NumOutputs();
    for (size_t i = 0; i < outputsNum; i++) {
      shared_ptr<OutputData> output =
          make_shared<OutputData>(runtimePtr->GetOutput(i), i , runtime_pipeline_output_conf);

      outputs->push_back(output);
    }

    rData.ExportAppendData(outputs);
    return;
  }
};

#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
