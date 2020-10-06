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

use std::{
    os::raw::{c_int, c_void},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread::{self, JoinHandle},
};

#[cfg(not(target_arch = "wasm32"))]
use std::env;

use crossbeam_channel::{bounded, Receiver, Sender};
use tvm_sys::ffi::TVMParallelGroupEnv;

pub(crate) type FTVMParallelLambda =
    extern "C" fn(task_id: usize, penv: *const TVMParallelGroupEnv, cdata: *const c_void) -> i32;

/// Holds a parallel job request made by a TVM library function.
struct Job {
    cb: FTVMParallelLambda,
    cdata: *const c_void,
    req_num_tasks: usize,
    pending: Arc<AtomicUsize>,
}

impl Job {
    /// Splits this job into a number of `Task`s which can be scheduled.
    fn tasks(&self, num_workers: usize) -> Vec<Task> {
        let num_tasks = if self.req_num_tasks == 0 {
            num_workers
        } else {
            self.req_num_tasks.min(num_workers)
        };
        self.pending.store(num_tasks, Ordering::SeqCst);

        let barrier = Arc::new(Barrier::new(num_tasks));

        (0..num_tasks)
            .map(move |i| Task {
                id: i,
                flambda: self.cb,
                penv: TVMParallelGroupEnv {
                    sync_handle: &Arc::clone(&barrier) as *const _ as *mut c_void,
                    num_task: num_tasks as i32,
                },
                cdata: self.cdata,
                pending: Arc::clone(&self.pending),
            })
            .collect()
    }

    /// Waits for all tasks in this `Job` to be completed.
    fn wait(&self) {
        while self.pending.load(Ordering::Acquire) > 0 {
            thread::yield_now();
        }
    }
}

/// A chunk of work requested by a TVM function.
struct Task {
    id: usize,
    flambda: FTVMParallelLambda,
    penv: TVMParallelGroupEnv,
    cdata: *const c_void,
    pending: Arc<AtomicUsize>,
}
unsafe impl Send for Task {}
unsafe impl Sync for Task {}

impl Task {
    fn run(self) -> i32 {
        let status = (self.flambda)(self.id, &self.penv as *const _, self.cdata);
        self.pending.fetch_sub(1, Ordering::AcqRel);
        status
    }
}

#[derive(Default)]
struct Threads {
    #[allow(unused)]
    handles: Vec<JoinHandle<()>>,
    queues: Vec<Sender<Task>>,
}

impl<'a> Threads {
    fn launch<F: Sync + Send + FnOnce(Receiver<Task>) + 'static + Copy>(
        num_threads: usize,
        cb: F,
    ) -> Self {
        let (handles, queues) = (0..num_threads)
            .map(|_| {
                let (p, c) = bounded(2);
                let handle = thread::spawn(move || cb(c.into()));
                (handle, p)
            })
            .unzip();
        Threads { handles, queues }
    }
}

struct ThreadPool {
    num_workers: usize,
    #[allow(unused)]
    threads: Threads,
}

thread_local!(static THREAD_POOL: ThreadPool = ThreadPool::new());

impl ThreadPool {
    fn new() -> Self {
        let num_workers = max_concurrency();
        ThreadPool {
            num_workers,
            threads: Threads::launch(num_workers, ThreadPool::run_worker),
        }
    }

    fn launch(&self, job: Job) {
        let mut tasks = job.tasks(self.num_workers + 1);

        for (i, task) in tasks.split_off(1).into_iter().enumerate() {
            self.threads.queues[i].send(task).expect("should send");
        }

        tasks.pop().unwrap().run();
        job.wait();
    }

    fn run_worker(queue: Receiver<Task>) {
        loop {
            let task = match queue.recv() {
                Ok(v) => v,
                Err(_) => break,
            };
            let result = task.run();
            if result == <i32>::min_value() {
                break;
            } else if result != 0 {
                panic!("Error running task.");
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn max_concurrency() -> usize {
    if let Ok(threads_str) = env::var("TVM_NUM_THREADS").or_else(|_| env::var("OMP_NUM_THREADS")) {
        if let Ok(threads) = usize::from_str_radix(&threads_str, 10) {
            return threads;
        }
    }
    num_cpus::get()
}

#[cfg(target_arch = "wasm32")]
fn max_concurrency() -> usize {
    0 // wasm doesn't support threads yet
}

#[no_mangle]
pub extern "C" fn TVMBackendParallelLaunch(
    cb: FTVMParallelLambda,
    cdata: *const c_void,
    num_task: usize,
) -> c_int {
    if max_concurrency() < 2 {
        let penv = TVMParallelGroupEnv {
            sync_handle: std::ptr::null_mut(),
            num_task: 1,
        };
        cb(0, &penv as *const _, cdata);
    } else {
        THREAD_POOL.with(|pool| {
            pool.launch(Job {
                cb,
                cdata,
                req_num_tasks: num_task,
                pending: Arc::new(AtomicUsize::new(0)),
            });
        });
    }
    0
}

// @see issue 988 for information on why this function is used.
#[no_mangle]
pub unsafe extern "C" fn TVMBackendParallelBarrier(
    _task_id: usize,
    penv: *const TVMParallelGroupEnv,
) {
    let barrier: &Arc<Barrier> = &*((*penv).sync_handle as *const Arc<Barrier>);
    barrier.wait();
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;

    #[test]
    fn test_max_concurrency() {
        env::set_var("TVM_NUM_THREADS", "42");
        env::set_var("OMP_NUM_THREADS", "24");
        assert_eq!(max_concurrency(), 42);
        env::remove_var("TVM_NUM_THREADS");
        assert_eq!(max_concurrency(), 24);
    }

    extern "C" fn _flambda(
        task_id: usize,
        penv: *const TVMParallelGroupEnv,
        cdata: *const c_void,
    ) -> i32 {
        if cdata.is_null() {
            return 0;
        }
        unsafe {
            let &(ref counter, ref task_ids_sum) = &*(cdata as *const (AtomicUsize, AtomicUsize));
            thread::sleep(Duration::from_millis(50 * task_id as u64));
            counter.fetch_add(1, Ordering::SeqCst);
            task_ids_sum.fetch_add(task_id, Ordering::SeqCst);
            assert_eq!((*penv).num_task, 3);
        }
        0
    }

    // #[test]
    // fn test_parallel_launch() {
    //     TVMBackendParallelLaunch(flambda, ptr::null(), 6);
    //     let counter = AtomicUsize::new(0);
    //     let task_ids_sum = AtomicUsize::new(0);
    //     let cdata = (counter, task_ids_sum);
    //     let num_tasks = 3;
    //     TVMBackendParallelLaunch(flambda, &cdata as *const _ as *const c_void, num_tasks);
    //     assert_eq!(cdata.0.load(Ordering::SeqCst), num_tasks);
    //     assert_eq!(
    //         cdata.1.load(Ordering::SeqCst),
    //         (0..num_tasks).sum::<usize>()
    //     );
    // }
}
