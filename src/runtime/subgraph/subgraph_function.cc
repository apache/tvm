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
#include "subgraph_function.h"

#include <utility>
using namespace tvm::runtime;

void subgraph_pipeline_run(const int& num, const shared_ptr<RuntimeItem>& curRunItem) {
  QUEUE* curQueue = curRunItem->queue;
  QUEUE* nextQueue = curRunItem->next->queue;

  /*
   * Wait at beginning, then only do wait once last time data poll failed,
   * the loop would break after an exit notification get received.
   */
  bool suc = false;
  while (curRunItem->waitPipeLineData(suc)) {
    suc = subgraph_queue_poll(curQueue, &curRunItem->rData);
    if (!suc) {
      continue;
    }

    curRunItem->Run();

    auto output = curRunItem->GetOutput();
    subgraph_queue_push(nextQueue, output);
    cout << num << " subgraph run..." << endl;
    curRunItem->notifyDataReadyToNext();
  }
  curRunItem->notifyNextExit();

  cout << "end " << __FUNCTION__ << " num " << num << endl;
}

thread* subgraph_pipeline_init(SHARED_RUNTIME_VEC* runtimes) {
  for (size_t i = 1; i < runtimes->size(); i++) {
    (*runtimes)[i]->t = move(thread(subgraph_pipeline_run, i, (*runtimes)[i]));
  }
  return NULL;
}

void subgraph_init(Array<Module> graphRuntimes, SHARED_RUNTIME_VEC* runtimes) {
  int len = graphRuntimes.size();
  for (int i = 0; i < len; i++) {
    QUEUE* sub_queue = createQueue<SLOT>(NULL, SUB_Q_SIZE);
    auto runItem = make_shared<RuntimeItem>(graphRuntimes[i], sub_queue);
    runtimes->push_back(runItem);
    /*
       set prev and next for RuntimeItem, runtime need these information to
       poll data from prev and do notification for next.
       */
    if (i > 0) {
      (*runtimes)[i - 1]->next = (*runtimes)[i];
    }
    if (i == len - 1) {
      (*runtimes)[i]->next = (*runtimes)[0];
    }
  }
#ifndef SERIALIZE
  subgraph_pipeline_init(runtimes);
#endif
  return;
}

inline void subgraph_queue_push(QUEUE* queue, Array<NDArray> arrays) {
  q_push<SLOT, Array<NDArray>>(queue, arrays);
  return;
}

bool subgraph_queue_poll(QUEUE* queue, RuntimeData* runtimeData) {
  return q_poll<SLOT, RuntimeData>(queue, runtimeData);
}

void subgraph_run_serial(const SHARED_RUNTIME_VEC runtimes) {
  runtimes[0]->Run();
  for (size_t i = 1; i < runtimes.size(); i++) {
    int oNum = runtimes[i - 1]->runtimePtr->NumOutputs();
    for (int j = 0; j < oNum; j++) {
      auto o = runtimes[i - 1]->runtimePtr->GetOutput(j);
      DLTensor* ptr = const_cast<DLTensor*>(o.operator->());
      runtimes[i]->runtimePtr->SetInput(j, ptr);
    }
    runtimes[i]->Run();
  }
}

void subgraph_run(const SHARED_RUNTIME_VEC& runtimes, bool synch) {
#ifdef SERIALIZE
  subgraph_run_serial(runtimes);
  return;
#endif
  shared_ptr<RuntimeItem> runtime = runtimes.front();
  runtime->Run();
  subgraph_queue_push(runtime->next->queue, runtime->GetOutput());
  runtime->notifyDataReadyToNext();
  return;
}

bool subgraph_poll(vector<NDArray>* output, const SHARED_RUNTIME_VEC& runtimes, const bool synch) {
  shared_ptr<RuntimeItem> firstRuntime = runtimes.front();
  QUEUE* queue = firstRuntime->queue;
#ifndef SERIALIZE
  bool suc = false;
  subgraphOutputData<> outputData(output);
  suc = q_poll<SLOT, subgraphOutputData<>>(queue, &outputData);
  if (!suc) {
    firstRuntime->waitPipeLineData(!synch);
    suc = q_poll<SLOT, subgraphOutputData<>>(queue, &outputData);
    cout << "run done suc is " << suc << endl;
  }
  return suc;
#else
  subgraphOutputData<> outputData(output);
  return q_poll<SLOT, subgraphOutputData<>>(queue, &outputData);
#endif
}

void subgraph_stop(const SHARED_RUNTIME_VEC& runtimes) {
  cout << __FUNCTION__ << endl;
  runtimes.front()->notifyNextExit();
}
