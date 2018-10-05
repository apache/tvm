use std::{
  os::raw::{c_int, c_void},
  sync::{
    atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT},
    Arc, Barrier,
  },
};

#[cfg(not(target_env = "sgx"))]
use num_cpus;
#[cfg(not(target_env = "sgx"))]
use std::{
  env,
  thread::{self, JoinHandle},
};

#[cfg(target_env = "sgx")]
use std::{collections::VecDeque, ptr, sync::Mutex};

use bounded_spsc_queue::{self, Producer};

use super::super::errors::*;
use ffi::runtime::TVMParallelGroupEnv;

#[cfg(target_env = "sgx")]
use super::{TVMArgValue, TVMRetValue};

type FTVMParallelLambda =
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
      }).collect()
  }

  /// Waits for all tasks in this `Job` to be completed.
  fn wait(&self) -> Result<()> {
    while self.pending.load(Ordering::Acquire) > 0 {
      #[cfg(not(target_env = "sgx"))]
      thread::yield_now();
    }
    Ok(())
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

impl FnOnce<()> for Task {
  type Output = i32;
  extern "rust-call" fn call_once(self, _args: ()) -> Self::Output {
    let status = (self.flambda)(self.id, &self.penv as *const _, self.cdata);
    self.pending.fetch_sub(1, Ordering::AcqRel);
    status
  }
}

#[derive(Default)]
struct Threads {
  #[allow(unused)]
  #[cfg(not(target_env = "sgx"))]
  handles: Vec<JoinHandle<()>>,
  queues: Vec<Producer<Task>>,
}

impl<'a> Threads {
  #[cfg(not(target_env = "sgx"))]
  fn launch<F: Sync + Send + FnOnce(Consumer<Task>) + 'static + Copy>(
    num_threads: usize,
    cb: F,
  ) -> Self {
    let (handles, queues) = (0..num_threads)
      .map(|_| {
        let (p, c) = bounded_spsc_queue::make(2);
        let handle = thread::spawn(move || cb(c.into()));
        (handle, p)
      }).unzip();
    Threads {
      handles: handles,
      queues: queues,
    }
  }

  #[cfg(target_env = "sgx")]
  fn launch<F: Sync + Send + FnOnce(Consumer<Task>) + 'static + Copy>(
    num_threads: usize,
    _cb: F,
  ) -> Self {
    let mut consumer_queues = SGX_QUEUES.lock().unwrap();
    let queues = (0..num_threads)
      .map(|_| {
        let (p, c) = bounded_spsc_queue::make(2);
        consumer_queues.push_back(c.into());
        p
      }).collect();
    ocall_packed!("__sgx_thread_group_launch__", num_threads as u64);
    Threads { queues: queues }
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
      num_workers: num_workers,
      threads: Threads::launch(num_workers, ThreadPool::run_worker),
    }
  }

  fn launch(&self, job: Job) {
    let mut tasks = job.tasks(self.num_workers + 1);

    for (i, task) in tasks.split_off(1).into_iter().enumerate() {
      self.threads.queues[i].push(task);
    }

    tasks.pop().unwrap()();
    job.wait().unwrap();
  }

  fn run_worker(queue: Consumer<Task>) {
    loop {
      let task = queue.pop();
      let result = task();
      if result == <i32>::min_value() {
        break;
      } else if result != 0 {
        panic!("Error running task.");
      }
    }
  }
}

// Send + Sync wrapper for bounded_spsc_queue::Consumer
struct Consumer<T> {
  consumer: bounded_spsc_queue::Consumer<T>,
}
impl<T> From<bounded_spsc_queue::Consumer<T>> for Consumer<T> {
  fn from(c: bounded_spsc_queue::Consumer<T>) -> Self {
    Consumer { consumer: c }
  }
}
impl<T> Consumer<T> {
  fn pop(&self) -> T {
    self.consumer.pop()
  }
}
unsafe impl<T> Send for Consumer<T> {}
unsafe impl<T> Sync for Consumer<T> {}

#[cfg(target_env = "sgx")]
lazy_static! {
  /// Holds tasks for untrusted threads which re-enter the enclave to execute.
  static ref SGX_QUEUES: Mutex<VecDeque<Consumer<Task>>> = Mutex::new(VecDeque::new());
}

#[cfg(all(not(target_arch = "wasm32"), not(target_env = "sgx")))]
fn max_concurrency() -> usize {
  if let Ok(threads_str) = env::var("TVM_NUM_THREADS").or(env::var("OMP_NUM_THREADS")) {
    if let Ok(threads) = usize::from_str_radix(&threads_str, 10) {
      return threads;
    }
  }
  num_cpus::get_physical()
}

#[cfg(target_env = "sgx")]
fn max_concurrency() -> usize {
  usize::from_str_radix(env!("TVM_NUM_THREADS"), 10).unwrap_or(1)
}

#[cfg(target_arch = "wasm32")]
fn max_concurrency() -> usize {
  0 // wasm doesn't support threads yet
}

#[cfg(target_env = "sgx")]
pub fn tvm_run_worker(_args: &[TVMArgValue]) -> TVMRetValue {
  let q = {
    let mut qs = SGX_QUEUES.lock().unwrap();
    qs.pop_front()
    // `qs: MutexGuard` needs to be dropped here since `run_worker` won't return
  };
  if let Some(q) = q {
    ThreadPool::run_worker(q);
  }
  TVMRetValue::default()
}

#[no_mangle]
pub extern "C" fn TVMBackendParallelLaunch(
  cb: FTVMParallelLambda,
  cdata: *const c_void,
  num_task: usize,
) -> c_int {
  if max_concurrency() == 0 {
    let penv = TVMParallelGroupEnv {
      sync_handle: 0 as *mut c_void,
      num_task: 1,
    };
    cb(0, &penv as *const _, cdata);
  } else {
    THREAD_POOL.with(|pool| {
      pool.launch(Job {
        cb: cb,
        cdata: cdata,
        req_num_tasks: num_task,
        pending: Arc::new(ATOMIC_USIZE_INIT),
      });
    });
  }
  return 0;
}

#[cfg(target_env = "sgx")]
pub(crate) fn sgx_join_threads() {
  extern "C" fn poison_pill(
    _task_id: usize,
    _penv: *const TVMParallelGroupEnv,
    _cdata: *const c_void,
  ) -> i32 {
    <i32>::min_value()
  }

  THREAD_POOL.with(|pool| {
    pool.launch(Job {
      cb: poison_pill,
      cdata: ptr::null(),
      req_num_tasks: 0,
      pending: Arc::new(ATOMIC_USIZE_INIT),
    });
  });
  ocall_packed!("__sgx_thread_group_join__", 0);
}

// @see https://github.com/dmlc/tvm/issues/988 for information on why this function is used.
#[no_mangle]
pub extern "C" fn TVMBackendParallelBarrier(_task_id: usize, penv: *const TVMParallelGroupEnv) {
  let barrier: &Arc<Barrier> = unsafe { &*((*penv).sync_handle as *const Arc<Barrier>) };
  barrier.wait();
}

#[cfg(test)]
mod tests {
  use std::{ptr, thread, time::Duration};

  use super::*;

  #[test]
  fn test_max_concurrency() {
    env::set_var("TVM_NUM_THREADS", "42");
    env::set_var("OMP_NUM_THREADS", "24");
    assert_eq!(max_concurrency(), 42);
    env::remove_var("TVM_NUM_THREADS");
    assert_eq!(max_concurrency(), 24);
  }

  extern "C" fn flambda(
    task_id: usize,
    penv: *const TVMParallelGroupEnv,
    cdata: *const c_void,
  ) -> i32 {
    if cdata == ptr::null() {
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

  #[test]
  fn test_parallel_launch() {
    TVMBackendParallelLaunch(flambda, ptr::null(), 6);
    let counter = ATOMIC_USIZE_INIT;
    let task_ids_sum = ATOMIC_USIZE_INIT;
    let cdata = (counter, task_ids_sum);
    let num_tasks = 3;
    TVMBackendParallelLaunch(flambda, &cdata as *const _ as *const c_void, num_tasks);
    assert_eq!(cdata.0.load(Ordering::SeqCst), num_tasks);
    assert_eq!(
      cdata.1.load(Ordering::SeqCst),
      (0..num_tasks).sum::<usize>()
    );
  }
}
